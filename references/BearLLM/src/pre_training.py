import torch
import os
from dotenv import dotenv_values
from tqdm import tqdm

from models.FCN import FaultClassificationNetwork as FCN

env = dotenv_values()
bearllm_weights = env.get('BEARLLM_WEIGHTS', './bearllm_weights')
fcn_weights = f'{bearllm_weights}/fcn'

# --- 动态选择数据集管道 ---
active_dataset = env.get('ACTIVE_DATASET', 'mbhm')
if active_dataset == 'cwru':
    from functions.cwru import get_loaders
    print(">>> 预训练模式：已加载 CWRU 数据集管道")
else:
    from functions.mbhm import get_loaders
    print(">>> 预训练模式：已加载 MBHM 数据集管道")

class HyperParameters:
    def __init__(self):
        # 针对 CWRU 数据集动态调整超参数
        if active_dataset == 'cwru':
            self.batch_size = 32      # CWRU 数据量较小，用 1024 会报错
            self.num_workers = 4
            self.lr_patience = 10     # 缩小 patience 加快学习率衰减
        else:
            self.batch_size = 1024
            self.num_workers = 10
            self.lr_patience = 150
            
        self.lr = 1e-4
        self.lr_factor = 0.5
        self.epoch_max = 50
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

class PreTrainner:
    def __init__(self):
        self.hp = HyperParameters()
        train_loader, val_loader, test_loader = get_loaders(self.hp.batch_size, self.hp.num_workers)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.model = FCN().to(self.hp.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.hp.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=self.hp.lr_patience,
                                                                    factor=self.hp.lr_factor)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.best_val_loss = 1e10
        self.best_val_acc = 0

    def train_epoch(self):
        self.model.train()
        for data, label in tqdm(self.train_loader):
            data = data.to(self.hp.device)
            label = label.to(self.hp.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, label)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step(loss)

    def eval_epoch(self):
        self.model.eval()
        with torch.no_grad():
            val_loss = 0
            correct = 0
            for i, (data, label) in enumerate(self.val_loader):
                data = data.to(self.hp.device)
                label = label.to(self.hp.device)
                output = self.model(data)
                loss = self.criterion(output, label)
                val_loss += loss.item()
                correct += (output.argmax(1) == label).sum().item()
            val_loss /= len(self.val_loader)
            val_acc = correct / len(self.val_loader.dataset)
        return val_loss, val_acc

    def test_epoch(self):
        self.model.eval()
        with torch.no_grad():
            correct = 0
            for i, (data, label) in enumerate(self.test_loader):
                data = data.to(self.hp.device)
                label = label.to(self.hp.device)
                output = self.model(data)
                correct += (output.argmax(1) == label).sum().item()
            test_acc = correct / len(self.test_loader.dataset)
        return test_acc

    def train(self):
        for epoch in range(self.hp.epoch_max):
            self.train_epoch()
            val_loss, val_acc = self.eval_epoch()
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.model.save_weights(fcn_weights)
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.model.save_weights(fcn_weights)
            print(f'epoch: {epoch}, val_loss: {val_loss}, val_acc: {val_acc}')
            if self.scheduler.state_dict()['_last_lr'][0] < 1e-7:
                break
        test_acc = self.test_epoch()
        print(f'test_acc: {test_acc}')

if __name__ == "__main__":
    pre_trainner = PreTrainner()
    pre_trainner.train()