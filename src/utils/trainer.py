import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

    def train(self, optimizer, epochs, pruner=None):
        history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'early_stop_iter': 0}
        best_val_loss = float('inf')
        total_iters = 0
        
        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                
                # Apply masks to gradients if pruner is provided
                if pruner:
                    for name, module in self.model.named_modules():
                        if name in pruner.masks:
                            module.weight.grad.mul_(pruner.masks[name])
                
                optimizer.step()
                running_loss += loss.item()
                total_iters += 1
            
            val_loss, val_acc = self.evaluate(self.val_loader)
            history['train_loss'].append(running_loss / len(self.train_loader))
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                history['early_stop_iter'] = total_iters
                
        return history

    def evaluate(self, loader):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        return running_loss / len(loader), correct / total
