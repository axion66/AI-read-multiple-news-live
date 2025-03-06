from flask import Flask, request, jsonify
from googlesearch import search
from flask_cors import CORS
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
import torch.nn as nn

app = Flask(__name__)
CORS(app)

class Finbert(nn.Module):
    def __init__(self):
        super().__init__()
        self.finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone', num_labels=3)
        self.tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
        self.nlp = pipeline("sentiment-analysis", model=self.finbert, tokenizer=self.tokenizer)

    def inference(self, text: list[str]):
        return self.nlp(text)

# Initialize FinBERT model
finbert_model = Finbert()

@app.route('/search')
def get_news():
    company = request.args.get('company', '')
    
    if not company:
        return jsonify({"error": "No company name provided"}), 400

    stock_results = search(f"{company} stock today", num_results=15, unique=True, advanced=True)
    news_results = search(f"{company} latest news", num_results=10, unique=True, advanced=True)

    def sentiment2num(txt):
        if txt=='Positive':
            return 1
        if txt=='Negative':
            return -1
        return 0
    def num2sentiment(num, ths=0.1):
        if num > ths:
            return "Positive"
        if num < -ths:
            return "Negative"
        return "Neutral"
    
    def parse_results(results):
        parsed_results = []
        sentiments = []
        for r in results:
            text = f"{r.title} {r.description}"
            sentiment = finbert_model.inference([text])[0]['label']
            sentiments.append(sentiment2num(sentiment)) 
            parsed_results.append({
                "title": r.title,
                "url": r.url,
                "description": r.description,
                "sentiment": sentiment 
            })
        average_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.0
        return parsed_results, average_sentiment

    latest_news, avg_news_sentiment = parse_results(news_results)
    stock_updates, avg_stock_sentiment = parse_results(stock_results)

    return jsonify({
        "latest_news": latest_news,
        "stock_updates": stock_updates,
        "average_sentiment": {
            "latest_news": num2sentiment(avg_news_sentiment),
            "stock_updates": num2sentiment(avg_stock_sentiment),
            "average" : num2sentiment(avg_news_sentiment*0.5 + avg_stock_sentiment*0.5)
        }
    })

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)

