import torch
import numpy as np
import scanpy as sc
import anndata
import sys
sys.path.append('..')
from src.models import scTransv2, MLPv2
from src.models.distillerv2 import run_distiller


def train_trans(ref_adata, epochs=3):
    batch_size=6

    mask=np.load("./result/mask.npy")
    print(mask.shape)
    scmodel=scTransv2.scTrans_model(num_classes=13, num_genes=3000, mask=mask, embed_dim=48, depth=1, num_heads=2, has_logits=False)
    print("model created")
    scmodel.compile(optimizer='sgd', sgd_epochs=epochs, lrf=0.01, lr=0.001)
    print("model compiled")
    scmodel.fit(ref_adata, label_name='Celltype',epochs=epochs, batch_size=batch_size)
    torch.save(scmodel.state_dict(), "./result/teacher_{}.pth".format(epochs))

ref_adata = sc.read('../data/demo_train.h5ad')
train_trans(ref_adata)