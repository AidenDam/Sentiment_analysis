import re
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from collections import Counter
import torch
from torch.utils.data import TensorDataset, DataLoader

from . import get_dataset

def preprocess_string(s):
    # Remove all non-word characters (everything except numbers and letters)
    s = re.sub(r"[^\w\s]", '', s)
    # Replace all runs of whitespaces with no space
    s = re.sub(r"\s+", '', s)
    # replace digits with no space
    s = re.sub(r"\d", '', s)

    return s

def fil_vocab(filename_vocab):
    word_list = []

    if filename_vocab is None:
        filename_vocab = get_dataset.filename_vocab
        with open(filename_vocab, 'r') as f:
            word_list = f.read().split('\n')
    else:
        stop_words = set(stopwords.words(get_dataset.filename_stop_words)) 
        with open(filename_vocab, 'r') as f:
            lines = f.read().split('\n')
            for sent in lines:
                for word in sent.lower().split():
                    word = preprocess_string(word)
                    if word not in stop_words and word != '':
                        word_list.append(word)

    return word_list

def one_hot_word(filename_vocab=None):
    word_list = fil_vocab(filename_vocab)

    corpus = Counter(word_list)
    # sorting on the basis of most common words
    corpus_ = sorted(corpus, key=corpus.get, reverse=True)[:1000]
    # creating a dict
    onehot_dict = {w: i+1 for i, w in enumerate(corpus_)}

    return onehot_dict

def tockenize(x_train, y_train, x_val, y_val, filename_vocab):
    onehot_dict = one_hot_word(filename_vocab)
  
    # tockenize
    final_list_train, final_list_test = [], []
    for sent in x_train:
        final_list_train.append([onehot_dict[preprocess_string(word)] for word in sent.lower().split() 
                                if preprocess_string(word) in onehot_dict.keys()])
    for sent in x_val:  
        final_list_test.append([onehot_dict[preprocess_string(word)] for word in sent.lower().split() 
                                if preprocess_string(word) in onehot_dict.keys()])
            
    return np.array(final_list_train, dtype=object), np.array(y_train), np.array(final_list_test, dtype=object), np.array(y_val), onehot_dict

def padding_(sentences, seq_len):
    features = np.zeros((len(sentences), seq_len),dtype=int)
    for ii, review in enumerate(sentences):
        if len(review) != 0:
            features[ii, -len(review):] = np.array(review)[:seq_len]
    return features

def read_data_train(filename_train, filename_val, filename_vocab=None, batch_size=64):
    df_train = pd.read_csv(filename_train)
    df_val = pd.read_csv(filename_val)

    x_train, y_train = df_train['sentence'], df_train['sentiment']
    x_val, y_val = df_val['sentence'], df_val['sentiment']

    x_train, y_train, x_val, y_val, vocab = tockenize(x_train, y_train, x_val, y_val, filename_vocab)

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

def get_model(model_name, num_layers, vocab_size, hidden_dim, embedding_dim, output_dim):
    if model_name == 'lstm':
        from models.lstm import SentimentRNN
        return SentimentRNN(num_layers, vocab_size, hidden_dim, embedding_dim, output_dim)
    elif model_name == 'birnn':
        from models.birnn import SentimentRNN
        return SentimentRNN(num_layers, vocab_size, hidden_dim, embedding_dim, output_dim)
    else:
        raise 'model must be lstm or birnn'

def predict_text(model, text, vocab, device='cpu'):
    word_seq = np.array([vocab[preprocess_string(word)] for word in text.split() 
                        if preprocess_string(word) in vocab.keys()])
    word_seq = np.expand_dims(word_seq, axis=0)
    pad =  torch.from_numpy(padding_(word_seq, 500))
    inputs = pad.to(device)
    output = model(inputs)

    return output.squeeze().argmax()