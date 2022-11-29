import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd

from utils.get_dataset import stop_words, get_dataset
from utils.utils import tockenize, padding_
from models.lstm import SentimentRNN

import argparse

parser = argparse.ArgumentParser(description='Choose option of data')
parser.add_argument('-d', '--data', type=str, default='full', help='small or full; default is full')
args = parser.parse_args()

device = ('cpu', 'cuda')[torch.cuda.is_available()]

if not os.path.exists('weight'):
    os.mkdir('weight')
file_name_model_save = 'weight/state_dict.pt'

batch_size = 50

def read_data(filename_train, filename_val, batch_size=50):
    df_train = pd.read_csv(filename_train)
    df_val = pd.read_csv(filename_val)

    x_train, y_train = df_train['sentence'], df_train['sentiment']
    x_val, y_val = df_val['sentence'], df_val['sentiment']

    x_train, y_train, x_val, y_val, vocab = tockenize(x_train, y_train, x_val, y_val, stop_words)

    # we have very less number of reviews with length > 500.
    # So we will consideronly those below it.
    x_train_pad = padding_(x_train, 500)
    x_val_pad = padding_(x_val, 500)

    # create Tensor datasets
    train_data = TensorDataset(torch.from_numpy(x_train_pad), torch.from_numpy(y_train))
    valid_data = TensorDataset(torch.from_numpy(x_val_pad), torch.from_numpy(y_val))

    # dataloaders

    # make sure to SHUFFLE your data
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)

    return train_loader, valid_loader, vocab

# function to predict accuracy
def acc(pred, label):
    return torch.sum(pred.squeeze().argmax(1) == label.squeeze()).item()

def train(model, train_loader, valid_loader, criterion, optimizer):
    clip = 5
    epochs = 4
    valid_loss_min = np.Inf
    # train for some number of epochs
    epoch_tr_loss, epoch_vl_loss = [], []
    epoch_tr_acc, epoch_vl_acc = [], []

    for epoch in range(epochs):
        train_losses = []
        train_acc = 0.0
        model.train()
        # initialize hidden state 
        h = None
        for inputs, labels in train_loader:
            if inputs.shape[0] != batch_size:
                continue
            
            inputs, labels = inputs.to(device), labels.to(device)   
            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            
            model.zero_grad()
            output, h = model(inputs, h)

            h = tuple([each.data for each in h])
            # calculate the loss and perform backprop
            loss = criterion(output.squeeze(), labels)
            loss.backward()
            train_losses.append(loss.item())
            # calculating accuracy
            accuracy = acc(output, labels)
            train_acc += accuracy
            #`clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

        val_h = None
        val_losses = []
        val_acc = 0.0
        model.eval()
        for inputs, labels in valid_loader:
            if inputs.shape[0] != batch_size:
                continue

            inputs, labels = inputs.to(device), labels.to(device)

            output, val_h = model(inputs, val_h)
            val_loss = criterion(output.squeeze(), labels)

            val_h = tuple([each.data for each in val_h])

            val_losses.append(val_loss.item())
            
            accuracy = acc(output, labels)
            val_acc += accuracy
                
        epoch_train_loss = np.mean(train_losses)
        epoch_val_loss = np.mean(val_losses)
        epoch_train_acc = train_acc/(len(train_loader.dataset)//batch_size*batch_size)
        epoch_val_acc = val_acc/(len(valid_loader.dataset)//batch_size*batch_size)
        epoch_tr_loss.append(epoch_train_loss)
        epoch_vl_loss.append(epoch_val_loss)
        epoch_tr_acc.append(epoch_train_acc)
        epoch_vl_acc.append(epoch_val_acc)
        print(f'Epoch {epoch+1}') 
        print(f'train_loss : {epoch_train_loss} val_loss : {epoch_val_loss}')
        print(f'train_accuracy : {epoch_train_acc*100} val_accuracy : {epoch_val_acc*100}')
        if epoch_val_loss <= valid_loss_min:
            torch.save(model.state_dict(), file_name_model_save)
            print('Validation loss decreased ({:.6f} --> {:.6f}). Saving model ...'.format(valid_loss_min, epoch_val_loss))
            valid_loss_min = epoch_val_loss
        print(25*'==')

if __name__ == '__main__':
    # get dataset
    assert args.data in ('small', 'full'), '-d must be small or full'

    filename_train, filename_val, filename_test = get_dataset(data=args.data, check_down=True)
    # end

    train_loader, valid_loader, vocab = read_data(filename_train, filename_val, batch_size)

    no_layers = 2
    vocab_size = len(vocab) + 1 # extra 1 for padding
    hidden_dim = 256
    embedding_dim = 64
    output_dim = 3
    lr=0.001

    model = SentimentRNN(no_layers, vocab_size, hidden_dim, embedding_dim, output_dim)
    # moving to gpu
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # train(model, train_loader, valid_loader, criterion, optimizer)