import omicverse as ov
import scanpy as sc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import scib
import matplotlib.pyplot as plt

adata = sc.read('./result_bmmc.h5ad')
print(adata)

print(adata.X)

true_label = adata.obs['cell type']
pred_label = adata.obs['Prediction']

accuracy = accuracy_score(true_label, pred_label)
precision = precision_score(true_label, pred_label, average="macro")
recall = recall_score(true_label, pred_label, average="macro")
macro_f1 = f1_score(true_label, pred_label, average="macro")
print("accuracy: ", accuracy)
print("precision: ", precision)
print("recall: ", recall)
print("macro_f1: ", macro_f1)

sc.pp.neighbors(adata, n_neighbors=10, use_rep="X_emb", metric="dice")
sc.tl.umap(adata)

sc.set_figure_params(dpi=80)
sc.pl.umap(adata, color=['cell type'])
plt.savefig("bmmc_true.png")

sc.pl.umap(adata, color=['Prediction'])
plt.savefig("bmmc_pred.png")

results = scib.metrics.metrics(
        adata,
        adata_int=adata,
        batch_key="batch",
        label_key="cell type",
        embed="X_emb",
        isolated_labels_asw_=False,
        silhouette_=False,
        hvg_score_=False,
        graph_conn_=False,
        pcr_=False,
        isolated_labels_f1_=False,
        trajectory_=False,
        nmi_=True,  # use the clustering, bias to the best matching
        ari_=True,  # use the clustering, bias to the best matching
        cell_cycle_=False,
        kBET_=False,  # kBET return nan sometimes, need to examine
        ilisi_=False,
        clisi_=False,
)

nmi = results[0].to_dict()['NMI_cluster/label']
ari = results[0].to_dict()['ARI_cluster/label']

print("nmi: ", nmi)
print("ari: ", ari)

print(results[0].to_dict())