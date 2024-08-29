import numpy as np
import scanpy as sc
import anndata

ref_adata = sc.read('../data/demo_train.h5ad')
ref_adata = ref_adata[:,ref_adata.var_names]
print(ref_adata)

query_adata = sc.read('../data/demo_test.h5ad')
query_adata = query_adata[:,query_adata.var_names]
print(query_adata)

print(ref_adata.var_names)
print(query_adata.var_names)