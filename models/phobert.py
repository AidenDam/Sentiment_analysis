import torch
from transformers import RobertaForSequenceClassification, AutoTokenizer

class PhoBert:
    def __init__(self) -> None:
        self.device = ('cpu', 'cuda')[torch.cuda.is_available()]
        self.model = RobertaForSequenceClassification.from_pretrained('AidenDam/PhoBERT_Sentiment_analysis', num_labels=3, cache_dir='weights/')
        self.tokenizer = AutoTokenizer.from_pretrained("AidenDam/PhoBERT_tokenizer_Sentiment_analysis", cache_dir='weights/')
        self.model.to(self.device)

    def predict(self, text: str) -> torch.Tensor:
        inp = self.tokenizer(text, return_tensors="pt")
        inp = {k: v.to(self.device) for k, v in inp.items()}

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inp)

        return outputs.logits