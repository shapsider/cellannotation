import os, sys

import anndata
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import torch

from utils import _utils



def load_train_adata(args):

    if args.anndata is not None:
        print("Load pre-processed anndata h5ad file..")
        train_adata = anndata.read_h5ad(args.anndata)
    else:
        if args.input is None or args.metadata is None:
            sys.exit(
                "Please make sure that both gene score matrix and metadata are provided!"
            )

        print(
            "Loading data... \n This may take a while depending on your data size.."
        )
        if '.csv' in args.input:
            train_adata = _utils._csv_data_loader(args.input)
        else:
            train_adata = _utils._COOmtx_data_loader(args.input)
        metadata = _utils._metadata_loader(args.metadata)
        if _utils.Celltype_COLUMN not in metadata.columns:
            sys.exit(
                "Column '%s' is not found in metadata. Please make sure to include cell type information in the metadata."
                % _utils.Celltype_COLUMN)

        common_cells = set(train_adata.obs_names).intersection(
            set(metadata.index))
        print("%d common cells found between input data and metadata." %
              len(common_cells))

        if len(common_cells) == 0:
            sys.exit(
                "No common cells are found between input data and metadata, please check your data!"
            )

        if len(common_cells) < 100:
            print("There are too few cells. Cellcano might not be accurate.")

        train_adata = train_adata[list(common_cells)]
        train_adata.obs = train_adata.obs.merge(metadata,
                                                left_on="barcode",
                                                right_index=True,
                                                how='left')

        train_adata = _utils._process_adata(train_adata, process_type='train')
        print("Data shape after processing: %d cells X %d genes" %
              (train_adata.shape[0], train_adata.shape[1]))
        train_adata = _utils._select_feature(train_adata,
                                             fs_method=args.fs,
                                             num_features=args.num_features)
        train_adata = _utils._scale_data(train_adata)  ## center-scale
        _utils._visualize_data(train_adata,
                               args.output_dir,
                               prefix=args.prefix)
        _utils._save_adata(train_adata, args.output_dir, prefix=args.prefix)
    return train_adata


def train_MLP(args):
    MLP_DIMS = _utils.MLP_DIMS 

    train_adata = load_train_adata(args)

    x_train = _utils._extract_adata(train_adata)
    enc = OneHotEncoder(handle_unknown='ignore')
    y_train = enc.fit_transform(train_adata.obs[[_utils.Celltype_COLUMN
                                                 ]]).toarray()
    print("Cell type categories: ", enc.categories_[0])

    mlp = _utils._init_MLP(x_train,
                           y_train,
                           dims=MLP_DIMS,
                           seed=_utils.RANDOM_SEED)
    mlp.compile()
    mlp.fit(x_train, y_train)
    model_save_dir = args.output_dir + os.sep + args.prefix + 'MLP_model'
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    torch.save(mlp, os.path.join(model_save_dir, 'model.pt'))

    train_adata.var.loc[:, ['mean', 'std']].to_csv(model_save_dir + os.sep +
                                                   "features.txt",
                                                   sep='\t')
    # 保存enc信息
    with open(model_save_dir + os.sep + "onehot_encoder.txt", 'w') as f:
        for idx, cat in enumerate(enc.categories_[0]):
            f.write('%d:%s\n' % (idx, cat))


def train_KD(args):
    teacher_MLP_DIMS = _utils.Teacher_DIMS  #if args.teacher_ns is None else args.teacher_ns
    student_MLP_DIMS = _utils.Student_DIMS  #if args.student_ns is None else args.student_ns

    train_adata = load_train_adata(args)

    x_train = _utils._extract_adata(train_adata)
    enc = OneHotEncoder(handle_unknown='ignore')
    y_train = enc.fit_transform(train_adata.obs[[_utils.Celltype_COLUMN
                                                 ]]).toarray()
    print("Cell type categories: ", enc.categories_)

    teacher = _utils._init_MLP(x_train,
                               y_train,
                               dims=teacher_MLP_DIMS,
                               seed=_utils.RANDOM_SEED)
    teacher.compile()
    teacher.fit(x_train, y_train, batch_size=_utils.BATCH_SIZE)
    student = _utils._init_MLP(x_train,
                               y_train,
                               dims=student_MLP_DIMS,
                               seed=_utils.RANDOM_SEED)
    distiller = _utils._run_distiller(x_train,
                                      y_train,
                                      student_model=student,
                                      teacher_model=teacher)
    save_path = args.output_dir + os.sep + args.prefix + 'KD_model'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(distiller, os.path.join(save_path, 'model.pt'))

    train_adata.var.loc[:, ['mean', 'std']].to_csv(save_path + os.sep +
                                                   "features.txt",
                                                   sep='\t')

    with open(save_path + os.sep + "onehot_encoder.txt", 'w') as f:
        for idx, cat in enumerate(enc.categories_[0]):
            f.write('%d:%s\n' % (idx, cat))
