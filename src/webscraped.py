import requests
import json
from dotenv import load_dotenv
from datetime import datetime, timedelta
try:
    load_dotenv()
except:
    print("You need to have API_KEY for newsapi.org in .env")
import os

def news1(company_name, n=10, duration=14):
    """
        company_name : Apple, Meta, etc
        n: Number of articles getting.
        duration: number of past {duration} days will be collected. duration=14 means getting news from the past 14 days.
    """
    api_key = os.getenv("API_KEY")
    
    past = (datetime.utcnow() - timedelta(days=duration)).strftime('%Y-%m-%dT%H:%M:%SZ')
    current = (datetime.utcnow() - timedelta(days=2)).strftime('%Y-%m-%dT%H:%M:%SZ')

    
    query = f"{company_name} company"
    url = f'https://newsapi.org/v2/everything?q={query}&from={past}&to={current}&sortBy=publishedAt&language=en&pageSize={n}&apiKey={api_key}'
    
    response = requests.get(url)
    
    if (response.status_code != 200):
        raise Exception("Failed to fetch news, HTTP status code:", response.status_code)
    news_data = response.json()
    if (news_data['status'] != 'ok'):
        raise Exception("Perhaps Token Limit!")
    process = [o for o in news_data['articles'][1:] if (company_name in o['content'] or company_name in o['description'] or company_name.lower() in o['description'] or company_name.lower() in o['content'])]
    
    text_infos = {
        'company_name' : query,
        # also 'articles'
        'articles': process
    }
    return text_infos




from pygooglenews import GoogleNews
import pandas as pd
from datetime import date, timedelta

def news2(company_name):
    today = date.today()
    yesterday = today - timedelta(days=1)
    now = today.strftime("%m/%d/%Y")
    yesterday = yesterday.strftime("%m/%d/%Y")

    googlenews = GoogleNews(start=yesterday, end=now)
    googlenews.search(company_name)
    result = googlenews.result()
    df = pd.DataFrame(result)
    return df

if __name__ == '__main__':
    company_name = input("Please provide the name of the Company or a Ticker: ")
    if company_name:
        print(f"Searching for and analyzing {company_name}, please be patient, it might take a while...")
        news_df = news2(company_name)
        print(news_df)
    else:
        print("No company name provided.")
        