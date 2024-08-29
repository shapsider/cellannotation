import numpy as np
from collections import OrderedDict
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
import pandas as pd


def todense(adata):
    import scipy
    if isinstance(adata.X, scipy.sparse.csr_matrix):
        return adata.X.todense()
    else:
        return adata.X

class MyDataSet(Dataset):

    def __init__(self, exp, label):
        self.exp = exp
        self.label = label
        self.len = len(label)
    def __getitem__(self,index):
        return self.exp[index],self.label[index]
    def __len__(self):
        return self.len

def balance_populations(data):
    ct_names = np.unique(data[:,-1])
    ct_counts = pd.value_counts(data[:,-1])
    max_val = min(ct_counts.max(),np.int32(2000000/len(ct_counts)))
    balanced_data = np.empty(shape=(1,data.shape[1]),dtype=np.float32)
    for ct in ct_names:
        tmp = data[data[:,-1] == ct]
        idx = np.random.choice(range(len(tmp)), max_val)
        tmp_X = tmp[idx]
        balanced_data = np.r_[balanced_data,tmp_X]
    return np.delete(balanced_data,0,axis=0)
  
def splitDataSet(adata, label_name='Celltype', tr_ratio= 0.7): 
    """ 
    Split data set into training set and test set.

    """
    label_encoder = LabelEncoder()
    el_data = pd.DataFrame(todense(adata),index=np.array(adata.obs_names).tolist(), columns=np.array(adata.var_names).tolist())
    el_data[label_name] = adata.obs[label_name].astype('str')
    #el_data = pd.read_table(data_path,sep=",",header=0,index_col=0)
    genes = el_data.columns.values[:-1]
    el_data = np.array(el_data)
    # el_data = np.delete(el_data,-1,axis=1)
    el_data[:,-1] = label_encoder.fit_transform(el_data[:,-1])
    inverse = label_encoder.inverse_transform(range(0,np.max(el_data[:,-1])+1))
    el_data = el_data.astype(np.float32)
    el_data = balance_populations(data = el_data)
    n_genes = len(el_data[1])-1
    train_size = int(len(el_data) * tr_ratio)
    train_dataset, valid_dataset = torch.utils.data.random_split(el_data, [train_size,len(el_data)-train_size])
    exp_train = torch.from_numpy(train_dataset.dataset[:,:n_genes].astype(np.float32))
    label_train = torch.from_numpy(train_dataset.dataset[:,-1].astype(np.int64))
    exp_valid = torch.from_numpy(valid_dataset.dataset[:,:n_genes].astype(np.float32))
    label_valid = torch.from_numpy(valid_dataset.dataset[:,-1].astype(np.int64))
    return exp_train, label_train, exp_valid, label_valid, inverse,genes

def generate_dataset(adata, batch_size=8, label_name='Celltype'):
    exp_train, label_train, exp_valid, label_valid, inverse, genes = splitDataSet(adata, label_name)

    train_dataset = MyDataSet(exp_train, label_train)
    valid_dataset = MyDataSet(exp_valid, label_valid)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,drop_last=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,drop_last=True)
    num_classes = np.int64(torch.max(label_train)+1)

    return train_loader, valid_loader

def statistic(adata, label_name='Celltype'):
    exp_train, label_train, exp_valid, label_valid, inverse, genes = splitDataSet(adata, label_name)
    num_classes = np.int64(torch.max(label_train)+1)
    num_genes = len(exp_train[0])
    return num_classes, num_genes

def extract_data_for_pred(adata): 
    return todense(adata)
