import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import sys
import numpy as np
import random

from src.models.dataset import generate_dataset, splitDataSet, MyDataSet
from src.models.lossv2 import SelfEntropyLoss, DDCLoss

def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)

class PyTorchDistiller:
    def __init__(self, student, teacher, loss_weights=[0.5, 0.5]):
        self.student = student
        self.teacher = teacher
        self.teacher.eval()  # Set teacher to evaluation mode
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.student.parameters())
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.student.to(self.device)
        self.teacher.to(self.device)
        self.random_seed = 0

        # 两个正则化损失
        self.sce = SelfEntropyLoss(loss_weights[0])
        self.ddc = DDCLoss(student.n_classes, loss_weights[1])

    def distillation_loss(self, student_logits, teacher_logits, temperature,
                          alpha):
        soft_loss = nn.KLDivLoss()(
            nn.functional.log_softmax(student_logits / temperature, dim=1),
            nn.functional.softmax(teacher_logits / temperature, dim=1))
        return alpha * soft_loss

    def fit(self, adata, label_name='Celltype', batch_size=8, epochs=10, alpha=0.1, temperature=3):
        GLOBAL_SEED = self.random_seed
        set_seed(GLOBAL_SEED)
        device = 'cuda'
        device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(device)
        # tb_writer = SummaryWriter()
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
            
        for epoch in range(epochs):
            train_loss, train_acc = self.train_one_epoch(data_loader=train_loader,
                                                         epoch=epoch)
            val_acc = self.evaluate(data_loader=valid_loader,
                                              epoch=epoch)
        print('Training finished!')


    def train_one_epoch(self, data_loader, epoch, alpha=0.1, temperature=3):
        self.student.train()
        self.optimizer.zero_grad()
        accu_loss = torch.zeros(1).to(self.device)
        
        accu_loss_sce = torch.zeros(1).to(self.device)
        accu_loss_ddc = torch.zeros(1).to(self.device)

        accu_num = torch.zeros(1).to(self.device)
        sample_num = 0
        data_loader = tqdm(data_loader)
        for step, data in enumerate(data_loader):
            batch_x, batch_y = data
            self.optimizer.zero_grad()
            batch_x = batch_x.to(self.device)

            batch_y = batch_y.to(self.device)
            sample_num += batch_x.shape[0]
            hidden, student_outputs = self.student(batch_x)
            _,pred,_ = self.teacher(batch_x)
            teacher_outputs = pred.detach(
            )
            pred_classes = torch.max(student_outputs, dim=1)[1]
            accu_num += torch.eq(pred_classes, batch_y).sum()

            loss_dist = self.criterion(
                student_outputs, batch_y) + self.distillation_loss(
                    student_outputs, teacher_outputs, temperature, alpha)

            loss_sce = self.sce(student_outputs)
            loss_ddc = self.ddc(hidden, student_outputs)

            loss = loss_dist + loss_sce + loss_ddc

            loss.backward()

            nn.utils.clip_grad_norm_(self.student.parameters(), 25)

            accu_loss += loss.detach()

            accu_loss_sce += loss_sce.detach()
            accu_loss_ddc += loss_ddc.detach()

            data_loader.desc = "[train epoch {}] loss: {:.3f}, sce loss: {:.3f}, ddc loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_loss_sce.item() / (step + 1),
                                                                               accu_loss_ddc.item() / (step + 1),
                                                                               accu_num.item() / sample_num)
            if not torch.isfinite(loss):
                print('WARNING: non-finite loss, ending training ', loss)
                sys.exit(1)
            self.optimizer.step()
        return accu_loss.item() / (step + 1), accu_num.item() / sample_num
    
    @torch.no_grad()
    def evaluate(self, data_loader, epoch):
        self.student.eval()
        accu_num = torch.zeros(1).to(self.device)
        sample_num = 0
        data_loader = tqdm(data_loader)
        for step, data in enumerate(data_loader):
            exp, labels = data
            sample_num += exp.shape[0]
            _, pred = self.student(exp.to(self.device))
            pred_classes = torch.max(pred, dim=1)[1]
            accu_num += torch.eq(pred_classes, labels.to(self.device)).sum()
            data_loader.desc = "[valid epoch {}] acc: {:.3f}".format(epoch, accu_num.item() / sample_num)
        return accu_num.item() / sample_num


def run_distiller(adata, 
                   student_model,
                   teacher_model, 
                   label_name='Celltype',
                   batch_size=8,
                   epochs=30,
                   alpha=0.1,
                   temperature=3):
    distiller = PyTorchDistiller(student=student_model, teacher=teacher_model)
    distiller.fit(adata=adata, label_name=label_name, batch_size=batch_size, epochs=epochs, alpha=alpha, temperature=temperature)
    return distiller.student