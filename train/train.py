import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model_architect import SimpleNN
from args_helper import argv_parser

## params
parser = argv_parser()
opt = parser.parse_args()

input_size = opt.input_size
output_size = opt.output_size
epochs = opt.epochs
batch_size = opt.batch_size
hidden_size = opt.hidden_size
ckpt_dir = opt.checkpoint_folder
model_dir = opt.model_dir

train_nums = 100
valid_nums = 50
device = "cuda" if torch.cuda.is_available() else "cpu"

## prepare training data
X_train = torch.rand(train_nums, input_size).to(device)
Y_train = torch.rand(train_nums, output_size).to(device)
X_valid = torch.rand(valid_nums, input_size).to(device)
Y_valid = torch.rand(valid_nums, output_size).to(device)

train_steps = int(np.ceil(X_train.shape[0] / batch_size))
valid_steps = int(np.ceil(X_valid.shape[0] / batch_size))

model = SimpleNN(input_size, hidden_size, output_size)
criterion = nn.MSELoss()  
optimizer = optim.SGD(model.parameters(), lr=1e-3)

model.to(device)
for ep in range(epochs):
    ## train models
    train_loss = 0
    model.train()
    for i in range(train_steps):
        x_batch = X_train[i*batch_size:(i+1)*batch_size]
        y_batch = Y_train[i*batch_size:(i+1)*batch_size]

        loss = model.update(x_batch, y_batch, optimizer, criterion)
        train_loss += loss

    train_loss /= train_steps
    print('EPOCH:%02d,TRAIN_LOSS:%.3f'%(ep+1, train_loss))

    ## validate model
    valid_rmse = 0
    model.eval()
    for i in range(valid_steps):
        x_batch = X_valid[i*batch_size:(i+1)*batch_size]
        y_batch = Y_valid[i*batch_size:(i+1)*batch_size]

        with torch.no_grad():
            y_pred = model(x_batch)

        ## compute rmse
        rmse = torch.sqrt(torch.mean((y_pred - y_batch)**2))
        rmse = rmse.cpu().numpy()

        valid_rmse += rmse

    valid_rmse /= valid_steps
    print('VALID LOSS:%.3f'%(valid_rmse))

    ## save models to model checkpoint
    torch.save(model.state_dict(), 
               f'{ckpt_dir}/ckpt_ep{ep+1:03d}_vrmse{valid_rmse:.3f}.pt')

## save final model to model_dir
torch.save(model.state_dict(), f'{model_dir}/final_model.pt')
