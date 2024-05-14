import torch
import torch.nn as nn
import torch.optim as optim

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  
        self.relu = nn.ReLU()                          
        self.fc2 = nn.Linear(hidden_size, output_size) 

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def update(self, x, y, optimizer, loss_fn):
        optimizer.zero_grad() 
        pred = self(x)
        loss = loss_fn(pred, y) 
        loss.backward()                    
        optimizer.step()

        return loss.detach().cpu().numpy()


