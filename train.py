import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.get_dataset import get_dataset
from utils.utils import read_data_train, get_model, predict_text

import argparse

parser = argparse.ArgumentParser(description='Choose option of data')
parser.add_argument('-d', '--data', type=str, default='full', help='small or full; default is full')
parser.add_argument('-m', '--model', type=str, default='birnn', help='lstm or birnn')
parser.add_argument('-ep', '--epochs', type=int, default=5, help='number of epochs')
parser.add_argument('-bz', '--batch_size', type=int, default=64, help='batch size')
args = parser.parse_args()

device = ('cpu', 'cuda')[torch.cuda.is_available()]

if not os.path.exists('weight'):
    os.mkdir('weight')
file_name_model_save = 'weight/state_dict.pt'

# function to predict accuracy
def acc(pred, label):
    return torch.sum(pred.squeeze().argmax(1) == label.squeeze()).item()

def train(model, train_loader, valid_loader, criterion, optimizer, batch_size=50, epochs=5, clip=5):
    valid_loss_min = np.Inf
    # train for some number of epochs
    epoch_tr_loss, epoch_vl_loss = [], []
    epoch_tr_acc, epoch_vl_acc = [], []

    for epoch in range(epochs):
        train_losses = []
        train_acc = 0.0
        model.train()
        for inputs, labels in tqdm(train_loader, desc='Train', leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            model.zero_grad()
            output = model(inputs)

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

        val_losses = []
        val_acc = 0.0
        model.eval()
        for inputs, labels in tqdm(valid_loader, desc='Valib', leave=False):
            inputs, labels = inputs.to(device), labels.to(device)

            output = model(inputs)
            val_loss = criterion(output.squeeze(), labels)

            val_losses.append(val_loss.item())
            
            accuracy = acc(output, labels)
            val_acc += accuracy
                
        epoch_train_loss = np.mean(train_losses)
        epoch_val_loss = np.mean(val_losses)
        epoch_train_acc = train_acc/len(train_loader.dataset)
        epoch_val_acc = val_acc/len(valid_loader.dataset)
        epoch_tr_loss.append(epoch_train_loss)
        epoch_vl_loss.append(epoch_val_loss)
        epoch_tr_acc.append(epoch_train_acc)
        epoch_vl_acc.append(epoch_val_acc)
        print(f'Epoch {epoch+1}') 
        print(f'train_loss: {epoch_train_loss}  val_loss: {epoch_val_loss}')
        print(f'train_accuracy: {epoch_train_acc*100}  val_accuracy: {epoch_val_acc*100}')

        if epoch_val_loss <= valid_loss_min:
            torch.save(model.state_dict(), file_name_model_save)
            print('Validation loss decreased ({:.6f} --> {:.6f}). Saving model ...'.format(valid_loss_min, epoch_val_loss))
            valid_loss_min = epoch_val_loss
        print(35*'==')

def test(model, filename_test):
    df_test = pd.read_csv(filename_test)

    model.eval()
    with torch.no_grad():
        pred = [predict_text(model, it, vocab, device) for it in df_test['sentence']]

    print('Test accuracy:', np.sum(df_test['sentiment'] == pred)/len(df_test['sentiment']))

if __name__ == '__main__':
    # get dataset
    assert args.data in ('small', 'full'), '-d must be small or full'

    filename_train, filename_val, filename_test = get_dataset(data=args.data, check_down=True)
    # end

    batch_size = args.batch_size
    train_loader, valid_loader, vocab = read_data_train(filename_train, filename_val, batch_size=batch_size)

    num_layers = 2
    vocab_size = len(vocab) + 1 # extra 1 for padding
    hidden_dim = 256
    embedding_dim = 64
    output_dim = 3
    lr = 0.001

    model = get_model(args.model, num_layers, vocab_size, hidden_dim, embedding_dim, output_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    epochs = args.epochs

    train(model, train_loader, valid_loader, criterion, optimizer, batch_size, epochs)

    test(model, filename_test)
    