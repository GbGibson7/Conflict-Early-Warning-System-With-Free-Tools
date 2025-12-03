"""
Free Data Sources for Conflict Early Warning System
100% Free APIs: NewsAPI, Reddit, GDELT
"""
import requests
import pandas as pd
import praw
import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import time
import numpy as np
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DataConfig:
    """Configuration for data collection"""
    newsapi_key: str = os.getenv('NEWS_API_KEY', '')
    reddit_client_id: str = os.getenv('REDDIT_CLIENT_ID', '')
    reddit_client_secret: str = os.getenv('REDDIT_CLIENT_SECRET', '')
    max_records: int = 100
    retention_days: int = 7

class FreeDataCollector:
    """Collect data from free APIs"""
    
    def __init__(self, config: DataConfig = None):
        self.config = config or DataConfig()
        self.setup_clients()
        
    def setup_clients(self):
        """Setup API clients"""
        # NewsAPI
        self.newsapi_base = "https://newsapi.org/v2"
        
        # Reddit
        if (self.config.reddit_client_id and 
            self.config.reddit_client_secret):
            self.reddit = praw.Reddit(
                client_id=self.config.reddit_client_id,
                client_secret=self.config.reddit_client_secret,
                user_agent=os.getenv('REDDIT_USER_AGENT', 'ConflictEarlyWarning/1.0')
            )
        else:
            self.reddit = None
            
        # GDELT (no API key needed)
        self.gdelt_base = "https://api.gdeltproject.org/api/v2"
        
    def fetch_newsapi(self, query: str = "conflict Kenya", days: int = 7) -> pd.DataFrame:
        """Fetch from NewsAPI"""
        if not self.config.newsapi_key:
            logger.warning("No NewsAPI key configured")
            return pd.DataFrame()
            
        try:
            url = f"{self.newsapi_base}/everything"
            params = {
                'q': query,
                'from': (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d'),
                'to': datetime.now().strftime('%Y-%m-%d'),
                'language': 'en',
                'pageSize': min(self.config.max_records, 100),
                'apiKey': self.config.newsapi_key,
                'sortBy': 'publishedAt'
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            articles = data.get('articles', [])
            
            records = []
            for article in articles:
                records.append({
                    'source': 'newsapi',
                    'title': article.get('title', ''),
                    'description': article.get('description', ''),
                    'content': article.get('content', ''),
                    'url': article.get('url', ''),
                    'published_at': article.get('publishedAt', ''),
                    'source_name': article.get('source', {}).get('name', ''),
                    'author': article.get('author', ''),
                    'query': query,
                    'collected_at': datetime.now().isoformat()
                })
            
            logger.info(f"Fetched {len(records)} articles from NewsAPI")
            return pd.DataFrame(records)
            
        except Exception as e:
            logger.error(f"NewsAPI error: {e}")
            return pd.DataFrame()
    
    def fetch_reddit(self, subreddits: List[str] = None, 
                    query: str = "conflict", 
                    limit: int = 50) -> pd.DataFrame:
        """Fetch from Reddit"""
        if not self.reddit:
            logger.warning("Reddit not configured")
            return pd.DataFrame()
            
        if subreddits is None:
            subreddits = ['worldnews', 'news', 'Kenya', 'Africa']
            
        records = []
        
        for subreddit_name in subreddits:
            try:
                subreddit = self.reddit.subreddit(subreddit_name)
                
                # Search for posts
                for submission in subreddit.search(
                    query=query,
                    time_filter='week',
                    limit=limit//len(subreddits),
                    sort='relevance'
                ):
                    records.append({
                        'source': 'reddit',
                        'subreddit': subreddit_name,
                        'title': submission.title,
                        'text': submission.selftext,
                        'url': f"https://reddit.com{submission.permalink}",
                        'created_at': datetime.fromtimestamp(submission.created_utc),
                        'score': submission.score,
                        'num_comments': submission.num_comments,
                        'author': str(submission.author),
                        'upvote_ratio': submission.upvote_ratio,
                        'collected_at': datetime.now().isoformat()
                    })
                    
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Reddit error for r/{subreddit_name}: {e}")
                continue
                
        logger.info(f"Fetched {len(records)} posts from Reddit")
        return pd.DataFrame(records)
    
    def fetch_gdelt(self, query: str = "conflict Kenya") -> pd.DataFrame:
        """Fetch from GDELT"""
        try:
            # GDELT Global Knowledge Graph
            params = {
                "query": f"{query} sourcelang:eng",
                "mode": "artlist",
                "format": "json",
                "maxrecords": min(self.config.max_records, 250),
                "startdatetime": (datetime.now() - timedelta(days=7)).strftime('%Y%m%d%H%M%S'),
                "enddatetime": datetime.now().strftime('%Y%m%d%H%M%S')
            }
            
            response = requests.get(
                f"{self.gdelt_base}/doc/doc",
                params=params,
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                articles = data.get('articles', [])
                
                records = []
                for article in articles:
                    records.append({
                        'source': 'gdelt',
                        'title': article.get('title', ''),
                        'url': article.get('url', ''),
                        'seendate': article.get('seendate', ''),
                        'language': article.get('language', ''),
                        'sourcecountry': article.get('sourcecountry', ''),
                        'collected_at': datetime.now().isoformat()
                    })
                
                logger.info(f"Fetched {len(records)} events from GDELT")
                return pd.DataFrame(records)
                
        except Exception as e:
            logger.error(f"GDELT error: {e}")
            
        return pd.DataFrame()
    
    def generate_synthetic_data(self, n_samples: int = 100) -> pd.DataFrame:
        """Generate synthetic data for demo/testing"""
        logger.info(f"Generating {n_samples} synthetic records")
        
        regions = ['Nairobi', 'Mombasa', 'Kisumu', 'Nakuru', 'Eldoret']
        event_types = ['Protest', 'Violence', 'Peace Talk', 'Election', 'Economic']
        severity_levels = ['Low', 'Medium', 'High', 'Critical']
        
        records = []
        start_date = datetime.now() - timedelta(days=30)
        
        for i in range(n_samples):
            event_date = start_date + timedelta(
                days=np.random.randint(0, 30),
                hours=np.random.randint(0, 24)
            )
            
            region = np.random.choice(regions)
            event_type = np.random.choice(event_types)
            severity = np.random.choice(severity_levels, p=[0.5, 0.3, 0.15, 0.05])
            
            # Generate realistic text
            if event_type == 'Protest':
                text = f"Protest in {region} regarding {np.random.choice(['rights', 'wages', 'elections'])}"
            elif event_type == 'Violence':
                text = f"Violent incident reported in {region}, police responding"
            elif event_type == 'Peace Talk':
                text = f"Peace negotiations ongoing in {region}"
            else:
                text = f"{event_type} event in {region}"
            
            records.append({
                'source': 'synthetic',
                'title': text,
                'text': f"{text}. Details emerging. Authorities monitoring situation.",
                'region': region,
                'event_type': event_type,
                'severity': severity,
                'date': event_date,
                'latitude': np.random.uniform(-4.0, 4.0),
                'longitude': np.random.uniform(33.0, 41.0),
                'confidence': np.random.uniform(0.5, 1.0),
                'collected_at': datetime.now().isoformat()
            })
        
        return pd.DataFrame(records)
    
    def collect_all(self) -> pd.DataFrame:
        """Collect from all sources"""
        logger.info("Starting data collection from all sources")
        
        all_data = []
        
        # NewsAPI
        news_data = self.fetch_newsapi()
        if not news_data.empty:
            all_data.append(news_data)
            
        # Reddit
        reddit_data = self.fetch_reddit()
        if not reddit_data.empty:
            all_data.append(reddit_data)
            
        # GDELT
        gdelt_data = self.fetch_gdelt()
        if not gdelt_data.empty:
            all_data.append(gdelt_data)
            
        # If no real data, use synthetic
        if not all_data:
            logger.warning("No real data collected, using synthetic")
            synth_data = self.generate_synthetic_data(50)
            all_data.append(synth_data)
        
        # Combine all data
        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            
            # Save to file
            os.makedirs('data/raw', exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"data/raw/collected_{timestamp}.csv"
            combined.to_csv(filename, index=False)
            
            logger.info(f"Collected {len(combined)} total records")
            logger.info(f"Saved to {filename}")
            
            return combined
            
        return pd.DataFrame()

def main():
    """Main function for testing"""
    collector = FreeDataCollector()
    data = collector.collect_all()
    
    if not data.empty:
        print("\n✅ Data Collection Complete!")
        print(f"Total records: {len(data)}")
        print("\nSources breakdown:")
        print(data['source'].value_counts())
        
        # Save sample for processing
        sample_file = "data/processed/latest_data.csv"
        data.to_csv(sample_file, index=False)
        print(f"\n💾 Sample saved to: {sample_file}")
    else:
        print("❌ No data collected")

if __name__ == "__main__":
    main()