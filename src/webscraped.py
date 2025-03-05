import requests
import json
from dotenv import load_dotenv
from datetime import datetime, timedelta
try:
    load_dotenv()
except:
    print("You need to have API_KEY for newsapi.org in .env")
import os

def get_latest_news(company_name, n=10, duration=14):
    """
        company_name : Apple, Meta, etc
        n: Number of articles getting.
        duration: number of past {duration} days will be collected. duration=14 means getting news from the past 14 days.
    """
    api_key = os.getenv("API_KEY")
    
    past = (datetime.utcnow() - timedelta(days=duration)).strftime('%Y-%m-%dT%H:%M:%SZ')
    current = (datetime.utcnow()).strftime('%Y-%m-%dT%H:%M:%SZ')
    
    query = f'{company_name} company'
    url = f'https://newsapi.org/v2/everything?q={query}&from={past}&to={current}&sortBy=publishedAt&language=en&pageSize={n}&apiKey={api_key}'
    
    response = requests.get(url)
    
    if (response.status_code != 200):
        raise Exception("Failed to fetch news, HTTP status code:", response.status_code)
    news_data = response.json()
    if (news_data['status'] != 'ok'):
        raise Exception("Perhaps Token Limit!")

    text_infos = {
        'company_name' : query,
        # also 'articles'
        'articles': news_data['articles'][1:]
    }
    return text_infos
if __name__ == "__main__":
    company_name = input("Enter the company name: ")
    get_latest_news(company_name)
