import scanpy as sc
import matplotlib.pyplot as plt
from matplotlib.pyplot import rc_context
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

attn_adata = sc.read("result_pancancer.h5ad")

true_label = attn_adata.obs['cell type']
pred_label = attn_adata.obs['Prediction']

accuracy = accuracy_score(true_label, pred_label)
precision = precision_score(true_label, pred_label, average="macro")
recall = recall_score(true_label, pred_label, average="macro")
macro_f1 = f1_score(true_label, pred_label, average="macro")
print("accuracy: ", accuracy)
print("precision: ", precision)
print("recall: ", recall)
print("macro_f1: ", macro_f1)

sc.set_figure_params(dpi=80)
sc.pl.umap(attn_adata, color=['cell type'])
plt.savefig("pancancer_true.png")

sc.pl.umap(attn_adata, color=['Prediction'])
plt.savefig("pancancer_pred.png")

sub = attn_adata[attn_adata.obs['major type']=='cDC',:]

sc.tl.pca(sub, svd_solver='arpack')
sc.pp.neighbors(sub, n_neighbors=4, n_pcs=20)

sc.tl.paga(sub, groups='cell type')

sc.set_figure_params(dpi=300)
sc.pl.paga(sub, color=['cell type'], frameon=False, fontsize=6, edge_width_scale=0.25, threshold=0.4, node_size_scale=3)

plt.savefig("paga.png", dpi=80)

sc.pp.normalize_total(attn_adata, target_sum=1e4)
sc.pp.log1p(attn_adata)
attn_adata.raw = attn_adata

sub = attn_adata[attn_adata.obs['major type']=='cDC',:]
sub = sub[sub.obs.tissue=='T',:]

ct_list = list(set(sub.obs['cell type']))
target_list = ['cDC2_FCN1', 'cDC2_IL1B', 'cDC3_LAMP3']
for ct in ct_list:
    if ct not in target_list:
        sub = sub[sub.obs['cell type']!=ct,:]

with rc_context({'figure.figsize': (3, 4)}):
    sc.pl.violin(sub, 'REACTOME_TOLL_RECEPTOR_CASCADES', groupby='cell type', 
                 stripplot=True, 
                 inner='box', 
                 xlabel='',
                 ylabel='TOLL RECEPTOR', 
                 rotation=90)
plt.savefig("cell_path.png", dpi=80)

sub = attn_adata[attn_adata.obs['cell type']=='cDC3_LAMP3',:]
sc.tl.rank_genes_groups(sub, 'cancer', method='wilcoxon')
txt = pd.DataFrame(sub.uns['rank_genes_groups']['names']).head(10)
with rc_context({'figure.figsize': (4, 4)}):
    sc.pl.violin(sub, 'REACTOME_MEMBRANE_TRAFFICKING', groupby='cancer', stripplot=True, inner='box', 
                 xlabel='',
                 ylabel='MEMBRANE TRAFFICKING', 
                 rotation=90)
plt.savefig("disease_path.png", dpi=80)