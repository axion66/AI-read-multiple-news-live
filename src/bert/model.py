from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
import torch.nn as nn

class Finbert(nn.Module):
    def __init__(self):
        super().__init__()
        self.finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone',num_labels=3)
        self.tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
        self.nlp = pipeline("sentiment-analysis", model=self.finbert, tokenizer=self.tokenizer)
        
        # need to implement another if use customly-trained tokenizer
    def infernece(self, text : list[str], enablePrint=False):
        '''
        sentences = ["there is a shortage of capital, and we need extra financing",  
                    "growth is strong and we have plenty of liquidity", 
                    "there are doubts about our finances", 
                    "profits are flat",
                    "It's positive and negative at the sametime"]
        '''
        results = self.nlp(text)
        if (enablePrint): print(results)
        return results