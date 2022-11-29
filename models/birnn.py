import torch
import torch.nn as nn

class SentimentRNN(nn.Module):
  def __init__(self, num_layers, vocab_size, hidden_dim, embedding_dim, output_dim, drop_prob=0.5):
    super(SentimentRNN,self).__init__()

    self.output_dim = output_dim
    self.hidden_dim = hidden_dim

    self.num_layers = num_layers
    self.vocab_size = vocab_size

    # embedding and LSTM layers
    self.embedding = nn.Embedding(vocab_size, embedding_dim)
    
    #lstm
    self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=self.hidden_dim,
                        num_layers=num_layers, batch_first=True)
    
    
    # dropout layer
    self.dropout = nn.Dropout(drop_prob)

    # linear layer
    self.fc = nn.Linear(self.hidden_dim*2, output_dim)
    
  def forward(self, x):
    # embeddings and lstm_out
    embeds = self.embedding(x)  # shape: B x S x Feature   since batch = True

    #print(embeds.shape)  #[50, 500, 1000]
    lstm_out, (h_n, c_n) = self.lstm(embeds)
    output_fw = h_n[-2, :, :]  
    output_bw = h_n[-1, :, :] 

    output = torch.cat([output_fw, output_bw], dim=-1) 
    out = self.fc(output)

    return nn.functional.log_softmax(out, dim=-1)