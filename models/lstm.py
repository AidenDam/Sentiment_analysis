import torch.nn as nn

class SentimentRNN(nn.Module):
  def __init__(self, no_layers, vocab_size, hidden_dim, embedding_dim, output_dim=3):
    super(SentimentRNN,self).__init__()

    self.output_dim = output_dim
    self.hidden_dim = hidden_dim

    self.no_layers = no_layers
    self.vocab_size = vocab_size

    # embedding and LSTM layers
    self.embedding = nn.Embedding(vocab_size, embedding_dim)
    
    # lstm
    self.lstm = nn.LSTM(input_size=embedding_dim,hidden_size=self.hidden_dim,
                        num_layers=no_layers, batch_first=True)
    
    # dropout layer
    self.dropout = nn.Dropout(0.3)

    # linear and sigmoid layer
    self.fc = nn.Linear(self.hidden_dim, output_dim)
      
  def forward(self, x, hidden = None):
    batch_size = x.size(0)
    # embeddings and lstm_out
    embeds = self.embedding(x)  # shape: B x S x Feature   since batch = True
    
    # print(embeds.shape)  # [50, 500, 1000]
    lstm_out, hidden = self.lstm(embeds, hidden)
    lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim) 
    
    # dropout and fully connected layer
    out = self.dropout(lstm_out)
    out = self.fc(out)

    # sigmoid function
    sig_out = nn.functional.log_softmax(out, 1)
    
    # reshape to be batch_size first
    sig_out = sig_out.view(batch_size, sig_out.shape[0]//batch_size, self.output_dim)

    sig_out = sig_out[:, -1, :] # get last batch of labels

    # return last sigmoid output and hidden state
    return sig_out, hidden