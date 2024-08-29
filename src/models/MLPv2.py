import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import sys, os

import random
import numpy as np

from src.models.dataset import generate_dataset, splitDataSet, MyDataSet


def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)

class MLP(nn.Module):
    def __init__(self, dims):
        super(MLP, self).__init__()
        self.dims = dims 
        self.input_shape = None  
        self.n_classes = None  
        self.layers = nn.ModuleList()  
        self.random_seed = 0 

    def init_MLP_model(self, dropout_rate=0.5):

        for i in range(len(self.dims)):
            self.layers.append(nn.Dropout(p=dropout_rate))  
            if i == 0:
                self.layers.append(nn.Linear(self.input_shape, self.dims[i]))
            else:
                self.layers.append(nn.Linear(self.dims[i - 1], self.dims[i]))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(self.dims[-1],
                                     self.n_classes))  

    def compile(self, optimizer='adam'):

        if optimizer == 'adam':
            self.optimizer = optim.Adam(self.parameters(), lr=1e-4)
            lr_schedule = optim.lr_scheduler.ExponentialLR(self.optimizer,
                                                           gamma=0.95)
            self.lr_scheduler = lr_schedule
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, hidden):

        for layer in self.layers[:-1]:
            hidden = layer(hidden)
        clus = self.layers[-1](hidden)
        return hidden, clus

    def fit_old(self,
            x_train,
            y_train,
            batch_size=32,
            max_epochs=100,
            sample_weight=None,
            class_weight=None):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        x_train = torch.tensor(x_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        dataset = torch.utils.data.TensorDataset(x_train, y_train)
        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  pin_memory=True,
                                                  num_workers=12)
        for epoch in range(max_epochs):
            epoch_loss = 0.0
            correct_predictions = 0
            total_predictions = 0
            for batch_x, batch_y in data_loader:

                self.train()
                self.optimizer.zero_grad()
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                outputs = self(batch_x)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                _, labels = torch.max(batch_y, 1)
                total_predictions += batch_y.size(0)
                correct_predictions += (predicted == labels).sum().item()

            self.lr_scheduler.step()
            epoch_accuracy = 100 * correct_predictions / total_predictions
            avg_loss = epoch_loss / len(data_loader)
            print(
                f"Epoch [{epoch+1}/{max_epochs}], Loss: {avg_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%"
            )

    def predict(self, x_test, device):

        self.eval()
        with torch.no_grad():

            x_test = torch.tensor(x_test, dtype=torch.float32).to(device)
            hidden, logits = self(x_test)
            probabilities = F.softmax(logits, dim=1)
        return hidden.cpu().numpy(), probabilities.cpu().numpy()
    
    def fit(self, adata, label_name='Celltype',
              batch_size=8, epochs= 10):
        GLOBAL_SEED = self.random_seed
        set_seed(GLOBAL_SEED)
        device = 'cuda'
        device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(device)
        exp_train, label_train, exp_valid, label_valid, inverse,genes = splitDataSet(adata,label_name)
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

        self.to(device)
        print('Model builded!')

        for epoch in range(epochs):
            train_loss, train_acc = self.train_one_epoch(optimizer=self.optimizer,
                                                         data_loader=train_loader,
                                                         device=device,
                                                         epoch=epoch)
            self.lr_scheduler.step() 
            val_loss, val_acc = self.evaluate(data_loader=valid_loader,
                                              device=device,
                                              epoch=epoch)
        print('Training finished!')

    def train_one_epoch(self, optimizer, data_loader, device, epoch):

        self.train()
        loss_function = self.criterion
        accu_loss = torch.zeros(1).to(device) 
        accu_num = torch.zeros(1).to(device)
        optimizer.zero_grad()
        sample_num = 0
        data_loader = tqdm(data_loader)
        for step, data in enumerate(data_loader):
            exp, label = data
            sample_num += exp.shape[0]
            pred = self(exp.to(device))
            pred_classes = torch.max(pred, dim=1)[1]
            accu_num += torch.eq(pred_classes, label.to(device)).sum()
            loss = loss_function(pred, label.to(device))
            loss.backward()
            accu_loss += loss.detach()
            data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                                accu_loss.item() / (step + 1),
                                                                                accu_num.item() / sample_num)
            if not torch.isfinite(loss):
                print('WARNING: non-finite loss, ending training ', loss)
                sys.exit(1)
            optimizer.step() 
            optimizer.zero_grad()
        return accu_loss.item() / (step + 1), accu_num.item() / sample_num
    
    @torch.no_grad()
    def evaluate(self, data_loader, device, epoch):
        self.eval()
        loss_function = self.criterion
        accu_num = torch.zeros(1).to(device)
        accu_loss = torch.zeros(1).to(device)
        sample_num = 0
        data_loader = tqdm(data_loader)
        for step, data in enumerate(data_loader):
            exp, labels = data
            sample_num += exp.shape[0]
            pred = self(exp.to(device))
            pred_classes = torch.max(pred, dim=1)[1]
            accu_num += torch.eq(pred_classes, labels.to(device)).sum()
            loss = loss_function(pred, labels.to(device))
            accu_loss += loss
            data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                                accu_loss.item() / (step + 1),
                                                                                accu_num.item() / sample_num)
        return accu_loss.item() / (step + 1), accu_num.item() / sample_num


def mlp_model(num_classes, num_genes, embed_dim=[64, 16]):
    mlp = MLP(embed_dim)
    mlp.input_shape = num_genes
    mlp.n_classes = num_classes
    mlp.init_MLP_model()
    return mlp
