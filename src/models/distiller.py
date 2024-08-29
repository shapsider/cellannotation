import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import sys

from src.models.dataset import generate_dataset, splitDataSet, MyDataSet
from src.models.train_trans import set_seed

class Distiller(nn.Module):
    def __init__(self, student, teacher):
        super(Distiller, self).__init__()
        self.teacher = teacher
        self.student = student

    def compile(self,
                optimizer,
                metrics,
                student_loss_fn,
                distillation_loss_fn,
                alpha=0.1,
                temperature=3):
        self.optimizer = optimizer(self.student.parameters())
        self.metrics = metrics
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    def forward(self, x):
        return self.student(x)

    def train_step(self, data):

        x, y = data
        self.optimizer.zero_grad()

        teacher_predictions = self.teacher(x).detach()

        student_predictions = self.student(x)

        student_loss = self.student_loss_fn(student_predictions, y)
        distillation_loss = self.distillation_loss_fn(
            F.softmax(teacher_predictions / self.temperature, dim=1),
            F.softmax(student_predictions / self.temperature, dim=1))
        loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss

        loss.backward()
        self.optimizer.step()

        metric_result = self.metrics(y, student_predictions)

        return {
            "student_loss": student_loss.item(),
            "distillation_loss": distillation_loss.item(),
            "metric": metric_result
        }

    def test_step(self, data):
        x, y = data
        with torch.no_grad():
            y_prediction = self.student(x)
            student_loss = self.student_loss_fn(y, y_prediction)
            metric_result = self.metrics(y, y_prediction)

        return {"student_loss": student_loss.item(), "metric": metric_result}


class PyTorchDistiller:
    def __init__(self, student, teacher):
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

    def distillation_loss(self, student_logits, teacher_logits, temperature,
                          alpha):
        soft_loss = nn.KLDivLoss()(
            nn.functional.log_softmax(student_logits / temperature, dim=1),
            nn.functional.softmax(teacher_logits / temperature, dim=1))
        return alpha * soft_loss

    def train_old(self, x_train, y_train, epochs, alpha, temperature):
        x_train = torch.tensor(x_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        dataset = torch.utils.data.TensorDataset(x_train, y_train)
        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=32,
                                                  shuffle=True,
                                                  pin_memory=True,
                                                  num_workers=12)
        for epoch in range(epochs):
            self.student.train()
            self.optimizer.zero_grad()

            epoch_loss = 0.0
            correct_predictions = 0
            total_predictions = 0
            for batch_x, batch_y in data_loader:
                self.optimizer.zero_grad()
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                student_outputs = self.student(batch_x)
                teacher_outputs = self.teacher(batch_x).detach(
                )  # Detach teacher outputs so no gradients are backpropagated
                loss = self.criterion(
                    student_outputs, batch_y) + self.distillation_loss(
                        student_outputs, teacher_outputs, temperature, alpha)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                # Calculate accuracy
                _, predicted = torch.max(student_outputs, 1)
                _, labels = torch.max(batch_y, 1)
                total_predictions += batch_y.size(0)
                correct_predictions += (predicted == labels).sum().item()
            epoch_accuracy = 100 * correct_predictions / total_predictions
            avg_loss = epoch_loss / len(data_loader)
            print(
                f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%"
            )

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
            val_loss, val_acc = self.evaluate(data_loader=valid_loader,
                                              epoch=epoch)
        print('Training finished!')


    def train_one_epoch(self, data_loader, epoch, alpha=0.1, temperature=3):
        self.student.train()
        self.optimizer.zero_grad()
        accu_loss = torch.zeros(1).to(self.device) 
        accu_num = torch.zeros(1).to(self.device)
        sample_num = 0
        data_loader = tqdm(data_loader)
        for step, data in enumerate(data_loader):
        # for batch_x, batch_y in data_loader:
            batch_x, batch_y = data
            self.optimizer.zero_grad()
            batch_x = batch_x.to(self.device)

            batch_y = batch_y.to(self.device)
            sample_num += batch_x.shape[0]
            student_outputs = self.student(batch_x)
            _,pred,_ = self.teacher(batch_x)
            teacher_outputs = pred.detach(
            )  # Detach teacher outputs so no gradients are backpropagated
            pred_classes = torch.max(student_outputs, dim=1)[1]
            accu_num += torch.eq(pred_classes, batch_y).sum()
            loss = self.criterion(
                student_outputs, batch_y) + self.distillation_loss(
                    student_outputs, teacher_outputs, temperature, alpha)
            loss.backward()
            accu_loss += loss.detach()
            data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
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
        accu_loss = torch.zeros(1).to(self.device)
        sample_num = 0
        data_loader = tqdm(data_loader)
        for step, data in enumerate(data_loader):
            exp, labels = data
            sample_num += exp.shape[0]
            pred = self.student(exp.to(self.device))
            pred_classes = torch.max(pred, dim=1)[1]
            accu_num += torch.eq(pred_classes, labels.to(self.device)).sum()
            loss = self.criterion(pred, labels.to(self.device))
            accu_loss += loss
            data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                                accu_loss.item() / (step + 1),
                                                                                accu_num.item() / sample_num)
        return accu_loss.item() / (step + 1), accu_num.item() / sample_num


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