import pandas as pd
import numpy as np
import scanpy as sc
import sys
sys.path.append('..')
from src.models.pathway import get_gmt, read_gmt, create_pathway_mask
from src.models.dataset import splitDataSet,MyDataSet

def generate_mask(adata, gmt_path='human_gobp', label_name='Celltype', mask_ratio=0.015, max_gs=300, max_g=300, n_unannotated=1):

    exp_train, label_train, exp_valid, label_valid, inverse,genes = splitDataSet(adata, label_name)
    print("splited")

    if gmt_path is None:
        mask = np.random.binomial(1,mask_ratio,size=(len(genes), max_gs))
        pathway = list()
        for i in range(max_gs):
            x = 'node %d' % i
            pathway.append(x)
        print('connection')

    else:
        if '.gmt' in gmt_path:
            gmt_path = gmt_path
        else:
            gmt_path = get_gmt(gmt_path)
        reactome_dict = read_gmt(gmt_path, min_g=0, max_g=max_g)
        mask,pathway = create_pathway_mask(feature_list=genes,
                                          dict_pathway=reactome_dict,
                                          add_missing=n_unannotated,
                                          fully_connected=True)
        pathway = pathway[np.sum(mask,axis=0)>4]
        mask = mask[:,np.sum(mask,axis=0)>4]
        #print(mask.shape)
        pathway = pathway[sorted(np.argsort(np.sum(mask,axis=0))[-min(max_gs,mask.shape[1]):])]
        mask = mask[:,sorted(np.argsort(np.sum(mask,axis=0))[-min(max_gs,mask.shape[1]):])]
        #print(mask.shape)
        print('Mask loaded!')
    np.save('./result/mask.npy',mask)
    pd.DataFrame(pathway).to_csv('./result/pathway.csv') 
    pd.DataFrame(inverse,columns=[label_name]).to_csv('./result/label_dictionary.csv', quoting=None)

ref_adata = sc.read("../data/demo_train.h5ad")
generate_mask(ref_adata)
mask = np.load("./result/mask.npy")
print(mask.shape)