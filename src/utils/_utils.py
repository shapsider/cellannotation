import os
import math
import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import scipy
import random
import torch
from torch import nn
from torch import optim

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})

from sklearn.preprocessing import OneHotEncoder

from typing import TypeVar
A = TypeVar('anndata')
ENC = TypeVar('OneHotEncoder')

from src.models.distiller import Distiller
from src.models.MLP import MLP

RANDOM_SEED = 1993
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

MLP_DIMS = Teacher_DIMS = Student_DIMS = [64, 16]
BATCH_SIZE = 32
Celltype_COLUMN = "celltype"
PredCelltype_COLUMN = "pred_celltype"
ENTROPY_QUANTILE = 0.4

if torch.cuda.is_available():
    print("Num GPUs Available: 1")
    device = torch.device("cuda:0")
else:
    print("Num GPUs Available: 0")
    device = torch.device("cpu")


def _csv_data_loader(csv_input: str) -> A:
    df = pd.read_csv(csv_input, index_col=0)
    obs = pd.DataFrame(data=df.columns, index=df.columns)
    obs.columns = ["barcode"]
    var = pd.DataFrame(data=df.index, index=df.index)
    var.columns = ['gene_symbols']
    adata = anndata.AnnData(X=df.T, obs=obs, var=var)
    adata.obs_names_make_unique(join="-")
    adata.var_names_make_unique(join="-")

    adata = adata[:, adata.var_names.notnull()]
    adata.var_names = [i.upper() for i in list(adata.var_names)]
    return adata


def _metadata_loader(metadata):
    metadata = pd.read_csv(metadata, index_col=0, sep=',')
    return metadata


def _process_adata(adata, process_type='train', celltype_label='celltype'):
    adata = adata[:, adata.var_names.notnull(
    )] 
    adata.var_names = [i.upper() for i in list(adata.var_names)
                       ]  

    
    adata.var_names_make_unique()
    adata.obs_names_make_unique()

    Gene1Pattern = "ERCC"
    Gene2Pattern = "MT-"
    id_tmp1 = np.asarray(
        [not str(name).startswith(Gene1Pattern) for name in adata.var_names],
        dtype=bool)
    id_tmp2 = np.asarray(
        [not str(name).startswith(Gene2Pattern) for name in adata.var_names],
        dtype=bool)
    id_tmp = np.logical_and(id_tmp1, id_tmp2)
    adata._inplace_subset_var(id_tmp)

    if adata.shape[0] < 3 or adata.shape[1] < 3:
        sys.exit("Error: too few genes or cells left to continue..")

    sc.pp.normalize_per_cell(adata, counts_per_cell_after=10000, min_counts=0)
    sc.pp.log1p(adata)

    if process_type == 'train':
        cells = adata.obs.dropna(subset=[celltype_label]).index.tolist()
        adata = adata[cells]
    return adata


def _select_feature(adata: A,
                    fs_method="F-test",
                    num_features: int = 3000) -> A:
    if fs_method == "noFS":
        print("Cellcano will not perform feature selection.\n")
        return adata
    else:
        if num_features > adata.shape[1]:
            print(
                "Number of features is larger than data. Cellcano will not perform feature selection.\n"
            )
            return adata

    if fs_method == "F-test":
        print("Use F-test to select features.\n")
        if scipy.sparse.issparse(adata.X) or \
                isinstance(adata.X, pd.DataFrame):
            tmp_data = adata.X.toarray()
        else:
            tmp_data = adata.X

        cell_annots = adata.obs[Celltype_COLUMN].tolist()
        uniq_celltypes = set(cell_annots)
        array_list = []
        for celltype in uniq_celltypes:
            idx = np.where(np.array(cell_annots) == celltype)[0].tolist()
            array_list.append(tmp_data[idx, :])
        F, p = scipy.stats.f_oneway(*array_list)
        F_updated = np.nan_to_num(F)
        sorted_idx = np.argsort(F_updated)[-num_features:]
        features = adata.var_names[sorted_idx].tolist()
        features.sort()
        adata = adata[:, features]

    if fs_method == "seurat":
        print("Use seurat in scanpy to select features.\n")
        sc.pp.highly_variable_genes(adata,
                                    n_top_genes=num_features,
                                    subset=True)
    return adata


def _scale_data(adata):
    '''Center scale
    '''
    adata_copy = sc.pp.scale(adata, zero_center=True, max_value=6, copy=True)
    return adata_copy


def _visualize_data(adata,
                    output_dir,
                    color_columns=["celltype"],
                    reduction="tSNE",
                    prefix="data"):
    '''Visualize data 

    ---
    Input:
        - reduction: tSNE or UMAP
        - color_columns: plot on categories
    '''
    sc.tl.pca(adata, random_state=RANDOM_SEED)

    if reduction == "tSNE":
        sc.tl.tsne(adata,
                   use_rep="X_pca",
                   learning_rate=300,
                   perplexity=30,
                   n_jobs=1,
                   random_state=RANDOM_SEED)
        sc.pl.tsne(adata, color=color_columns)
        plt.tight_layout()
        plt.savefig(output_dir + os.sep + prefix + "tSNE_cluster.png")
    if reduction == "UMAP":
        sc.pp.neighbors(adata,
                        n_neighbors=20,
                        use_rep="X_pca",
                        random_state=RANDOM_SEED)
        sc.tl.umap(adata, random_state=RANDOM_SEED)
        sc.pl.umap(adata, color=color_columns)
        plt.tight_layout()
        plt.savefig(output_dir + os.sep + prefix + "umap_cluster.png")


def _save_adata(adata, output_dir, prefix=""):
    '''Save anndata as h5ad
    '''
    adata.write(output_dir + os.sep + prefix + 'adata.h5ad')


def _prob_to_label(y_pred: np.ndarray, encoders: dict) -> list:
    '''Turn predicted probabilites to labels
    --- 
    Input:
        - y_pred: Predicted probabilities
        - encoders: dictionary with mapping information
    ---
    Output:
        - a list containing predicted cell types
    '''
    pred_labels = y_pred.argmax(1)
    pred_celltypes = [encoders[label] for label in pred_labels]
    print("=== Predicted celltypes: ", set(pred_celltypes))
    return pred_celltypes


def _label_to_onehot(labels: list, encoders: dict) -> np.ndarray:
    '''Turn predicted labels to onehot encoder
    ---
    Input: 
        - labels: the input predicted cell types
        - encoders: dictionary with mapping information
    '''
    inv_enc = {v: k for k, v in encoders.items()}
    onehot_arr = np.zeros((len(labels), len(encoders)))
    pred_idx = [inv_enc[l] for l in labels]
    onehot_arr[np.arange(len(labels)), pred_idx] = 1
    return onehot_arr


def _extract_adata(adata: A) -> np.ndarray:
    '''Extract adata.X to a numpy array
    ---
    Output:
         - matrix in np.ndarray format
    '''
    if scipy.sparse.issparse(
            adata.X) or isinstance(adata.X, pd.DataFrame) or isinstance(
                adata.X, anndata._core.views.ArrayView):
        X = adata.X.toarray()
    else:
        X = adata.X
    return X


def _init_MLP(x_train, y_train, dims=[64, 16], seed=0):
    '''Initialize MLP model based on input data
    '''
    mlp = MLP(dims)
    mlp.input_shape = x_train.shape[1]
    #mlp.n_classes = len(set(y_train.argmax(1)))
    mlp.n_classes = y_train.shape[1]
    mlp.random_state = seed
    mlp.init_MLP_model()  ## init the model
    return mlp


def _select_confident_cells(adata, celltype_col):
    '''Select low entropy cells from each predicted cell type
    ---
    Input:
        - adata: anndata object
        - celltype_col: the column indicator
    '''
    low_entropy_cells = []
    for celltype in set(adata.obs[celltype_col]):
        celltype_df = adata.obs[adata.obs[celltype_col] == celltype]
        entropy_cutoff = np.quantile(celltype_df['entropy'],
                                     q=ENTROPY_QUANTILE)
        ## change to < instead of <= to deal with ties
        cells = celltype_df.index[np.where(
            celltype_df['entropy'] <= entropy_cutoff)[0]].tolist()
        num_cells = math.ceil(ENTROPY_QUANTILE * celltype_df.shape[0])
        if len(cells) > num_cells:
            random.seed(RANDOM_SEED)
            selected_cells = random.sample(cells, num_cells)
        else:
            selected_cells = cells
        low_entropy_cells.extend(selected_cells)
    high_entropy_cells = list(set(adata.obs_names) - set(low_entropy_cells))
    adata.obs.loc[low_entropy_cells, 'entropy_status'] = "low"
    adata.obs.loc[high_entropy_cells, 'entropy_status'] = "high"
    return adata


def _oversample_cells(adata, celltype_col):
    '''Oversample cell types with number of cells lower than average
    ---
    Input:
        - adata: anndata object from second round
        - celltype_col: the column indicator
    '''
    sampled_cells = []
    avg_cellnums = math.ceil(adata.shape[0] /
                             len(set(adata.obs[celltype_col])))
    for celltype in set(adata.obs[celltype_col]):
        celltype_df = adata.obs[adata.obs[celltype_col] == celltype]
        random.seed(RANDOM_SEED)
        if celltype_df.shape[0] < avg_cellnums:
            selected_cells = random.choices(list(celltype_df.index),
                                            k=avg_cellnums)
        else:
            selected_cells = list(celltype_df.index)
        sampled_cells.extend(selected_cells)
    sampled_adata = adata[sampled_cells]
    return sampled_adata.copy()


class PyTorchDistiller:
    def __init__(self, student, teacher):
        self.student = student
        self.teacher = teacher
        self.teacher.eval()  # Set teacher to evaluation mode
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.student.parameters())
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.student.to(self.device)
        self.teacher.to(self.device)

    def distillation_loss(self, student_logits, teacher_logits, temperature,
                          alpha):
        soft_loss = nn.KLDivLoss()(
            nn.functional.log_softmax(student_logits / temperature, dim=1),
            nn.functional.softmax(teacher_logits / temperature, dim=1))
        return alpha * soft_loss

    def train(self, x_train, y_train, epochs, alpha, temperature):
        x_train = torch.tensor(x_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        dataset = torch.utils.data.TensorDataset(x_train, y_train)
        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=32,
                                                  shuffle=True,
                                                  pin_memory=True,
                                                  num_workers=12)
        for epoch in range(epochs):
            self.student.train()
            self.optimizer.zero_grad()

            epoch_loss = 0.0
            correct_predictions = 0
            total_predictions = 0
            for batch_x, batch_y in data_loader:
                self.optimizer.zero_grad()
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                student_outputs = self.student(batch_x)
                teacher_outputs = self.teacher(batch_x).detach(
                )  # Detach teacher outputs so no gradients are backpropagated
                loss = self.criterion(
                    student_outputs, batch_y) + self.distillation_loss(
                        student_outputs, teacher_outputs, temperature, alpha)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                # Calculate accuracy
                _, predicted = torch.max(student_outputs, 1)
                _, labels = torch.max(batch_y, 1)
                total_predictions += batch_y.size(0)
                correct_predictions += (predicted == labels).sum().item()
            epoch_accuracy = 100 * correct_predictions / total_predictions
            avg_loss = epoch_loss / len(data_loader)
            print(
                f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%"
            )


def _run_distiller(x_train,
                   y_train,
                   student_model,
                   teacher_model,
                   epochs=30,
                   alpha=0.1,
                   temperature=3):
    distiller = PyTorchDistiller(student=student_model, teacher=teacher_model)
    distiller.train(x_train, y_train, epochs, alpha, temperature)
    return distiller.student
