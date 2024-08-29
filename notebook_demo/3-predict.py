import torch
import numpy as np
import scanpy as sc
import anndata
import sys
sys.path.append('..')
from src.models import scTransv2
from src.models.attn_trans import prediect
import matplotlib.pyplot as plt

def pred_trans(query_adata):
    mask=np.load("./result/mask.npy")
    scmodel=scTransv2.scTrans_model(num_classes=13, num_genes=3000, mask=mask, embed_dim=48, depth=1, num_heads=2, has_logits=False)
    print("model created")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    scmodel.to(device=device)
    scmodel.load_state_dict(torch.load("./result/teacher_3.pth", 
                                    map_location=device
                                    ), 
                        strict=False
                        )
    print("param loaded")

    new, gene2token = prediect(query_adata, 
            scmodel, 
            laten=True, 
            pathway="./result/pathway.csv",
            diction="./result/label_dictionary.csv",
            save_att = 'X_att', 
            save_lantent = 'X_lat',
            n_step=10000,
            cutoff=0.1,
            n_unannotated = 1,
            batch_size = 50,
            embed_dim=48)

    return new, gene2token

flag = "step2"

if flag == "step1":
    query_adata = sc.read('../data/demo_test.h5ad')
    new, gene2token = pred_trans(query_adata)
    print(new)
    new.write('./result/scTrans_attn.h5ad')
    
elif flag == "step2":
    attn_adata = sc.read('./result/scTrans_attn.h5ad')
    print(attn_adata.obs_names)
    query_adata = sc.read('../data/demo_test.h5ad')
    print(query_adata.obs_names)
    
    attn_adata.obsm['X_emb'] = attn_adata.X
    attn_adata.obs["Celltype"] = query_adata.obs["Celltype"]
    
    sc.pp.neighbors(attn_adata, n_neighbors=10, use_rep="X_emb", metric="cosine")
    sc.tl.umap(attn_adata)
    sc.pl.umap(attn_adata, color=['Celltype', 'Prediction'])
    plt.savefig("./result/umap.png")