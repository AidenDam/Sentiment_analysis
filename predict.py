import torch
from models.phobert import PhoBert

torch.set_grad_enabled(False)

model = PhoBert()
labels = ('negative', 'neutral', 'positive')

def predict(text: str):
    from utils.utils import get_model, one_hot_word, predict_text
    device = ('cpu', 'cuda')[torch.cuda.is_available()]
    vocab = one_hot_word()
    num_layers = 2
    vocab_size = len(vocab) + 1 # extra 1 for padding
    hidden_dim = 256
    embedding_dim = 64
    output_dim = 3
    model = get_model(args.model, num_layers, vocab_size, hidden_dim, embedding_dim, output_dim).to(device)
    model.load_state_dict(torch.load(args.weight, map_location=torch.device(device)))
    return labels[predict_text(model, text, vocab, device)]

def phobeart_predict(text: str):
    return labels[torch.argmax(model.predict(text), dim=-1)]

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Choose option of data')
    parser.add_argument('-m', '--model', type=str, default='birnn', help='lstm or birnn')
    parser.add_argument('-w', '--weight', type=str, default='weight/state_dict.pt', help='weight of model')
    args = parser.parse_args()

    text = 'Cụng tạm thôi :))'
    print(phobeart_predict(text))