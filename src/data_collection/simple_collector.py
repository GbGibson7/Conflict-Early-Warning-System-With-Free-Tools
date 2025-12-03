"""
Simple Data Collector - FIXED VERSION
Includes proper risk_level assignment
"""
import requests
import pandas as pd
from datetime import datetime, timedelta
import json
import os
import time
import numpy as np
import logging
from typing import List, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleDataCollector:
    """Collect data without Reddit API - FIXED"""
    
    def __init__(self):
        self.newsapi_key = os.getenv('NEWS_API_KEY', '')
        self.use_newsapi = bool(self.newsapi_key and 'demo' not in self.newsapi_key)
        
        # Free APIs (no keys needed)
        self.gdelt_url = "https://api.gdeltproject.org/api/v2/doc/doc"
        self.wikipedia_api = "https://en.wikipedia.org/w/api.php"
        
        # Conflict regions in Kenya
        self.kenya_regions = [
            'Nairobi', 'Mombasa', 'Kisumu', 'Nakuru', 'Eldoret',
            'Kakamega', 'Kisii', 'Meru', 'Thika', 'Malindi'
        ]
        
        # Risk keywords mapping
        self.risk_keywords = {
            'Critical': ['terror', 'explosion', 'death', 'kill', 'attack', 'crisis', 'emergency'],
            'High': ['violence', 'clash', 'riot', 'protest', 'unrest', 'tension'],
            'Medium': ['demonstration', 'march', 'strike', 'dispute', 'conflict'],
            'Low': ['peace', 'calm', 'normal', 'stable', 'quiet']
        }
    
    def assign_risk_level(self, text: str) -> str:
        """Assign risk level based on keywords in text"""
        text_lower = text.lower()
        
        for risk_level, keywords in self.risk_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return risk_level
        
        # Default to Medium if no keywords found
        return 'Medium'
    
    def fetch_newsapi_safe(self, query: str = "Kenya conflict", days: int = 7) -> pd.DataFrame:
        """Fetch from NewsAPI with error handling"""
        if not self.use_newsapi:
            return pd.DataFrame()
        
        try:
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': query,
                'from': (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d'),
                'to': datetime.now().strftime('%Y-%m-%d'),
                'language': 'en',
                'pageSize': 50,  # Free tier limit
                'apiKey': self.newsapi_key,
                'sortBy': 'publishedAt'
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            articles = data.get('articles', [])
            
            records = []
            for article in articles:
                title = article.get('title', '')
                description = article.get('description', '')
                full_text = f"{title} {description}"
                
                records.append({
                    'source': 'newsapi',
                    'title': title,
                    'description': description,
                    'content': article.get('content', '')[:500] if article.get('content') else '',
                    'url': article.get('url', ''),
                    'published_at': article.get('publishedAt', ''),
                    'source_name': article.get('source', {}).get('name', ''),
                    'author': article.get('author', ''),
                    'query': query,
                    'collected_at': datetime.now().isoformat(),
                    'region': self._extract_region(full_text),
                    'risk_level': self.assign_risk_level(full_text),
                    'text': full_text,
                    'has_conflict_keywords': 1 if any(kw in full_text.lower() for kw in 
                                                     self.risk_keywords['High'] + self.risk_keywords['Critical']) else 0
                })
            
            logger.info(f"📰 NewsAPI: Fetched {len(records)} articles")
            return pd.DataFrame(records)
            
        except Exception as e:
            logger.warning(f"NewsAPI unavailable: {e}")
            return pd.DataFrame()
    
    def fetch_gdelt_free(self, query: str = "Kenya") -> pd.DataFrame:
        """Fetch from GDELT (completely free, no API key)"""
        try:
            # Simple GDELT query for Kenya
            params = {
                "query": f"{query} sourcelang:eng",
                "mode": "artlist",
                "format": "json",
                "maxrecords": 100,
                "startdatetime": (datetime.now() - timedelta(days=7)).strftime('%Y%m%d%H%M%S'),
                "enddatetime": datetime.now().strftime('%Y%m%d%H%M%S')
            }
            
            response = requests.get(self.gdelt_url, params=params, timeout=60)
            
            if response.status_code == 200:
                data = response.json()
                articles = data.get('articles', [])
                
                records = []
                for article in articles:
                    title = article.get('title', '')
                    records.append({
                        'source': 'gdelt',
                        'title': title,
                        'description': title,
                        'content': '',
                        'url': article.get('url', ''),
                        'published_at': article.get('seendate', ''),
                        'source_name': 'GDELT',
                        'author': '',
                        'query': query,
                        'collected_at': datetime.now().isoformat(),
                        'region': self._extract_region(title),
                        'risk_level': self.assign_risk_level(title),
                        'text': title,
                        'has_conflict_keywords': 1 if any(kw in title.lower() for kw in 
                                                         self.risk_keywords['High'] + self.risk_keywords['Critical']) else 0
                    })
                
                logger.info(f"🌍 GDELT: Fetched {len(records)} events")
                return pd.DataFrame(records)
                
        except Exception as e:
            logger.warning(f"GDELT unavailable: {e}")
        
        return pd.DataFrame()
    
    def _extract_region(self, text: str) -> str:
        """Extract Kenyan region from text"""
        text_lower = text.lower()
        for region in self.kenya_regions:
            if region.lower() in text_lower:
                return region
        return np.random.choice(self.kenya_regions)
    
    def generate_smart_synthetic(self, n_samples: int = 100) -> pd.DataFrame:
        """Generate realistic synthetic conflict data with proper columns"""
        logger.info(f"🤖 Generating {n_samples} synthetic records")
        
        conflict_types = {
            'Protest': {'risk': 'Medium', 'sentiment': -0.4, 'conflict': 0.5},
            'Violence': {'risk': 'High', 'sentiment': -0.8, 'conflict': 0.8},
            'Peace Talk': {'risk': 'Low', 'sentiment': 0.3, 'conflict': 0.2},
            'Election': {'risk': 'Medium', 'sentiment': -0.2, 'conflict': 0.4},
            'Economic': {'risk': 'Low', 'sentiment': -0.1, 'conflict': 0.3},
            'Security': {'risk': 'High', 'sentiment': -0.6, 'conflict': 0.7}
        }
        
        records = []
        start_date = datetime.now() - timedelta(days=30)
        
        for i in range(n_samples):
            region = np.random.choice(self.kenya_regions)
            conflict_type = np.random.choice(list(conflict_types.keys()))
            details = conflict_types[conflict_type]
            
            # Generate realistic text
            if conflict_type == 'Protest':
                text = f"Protests in {region} over {np.random.choice(['rights', 'wages', 'elections'])}"
            elif conflict_type == 'Violence':
                text = f"Violent clashes reported in {region}, emergency response deployed"
            elif conflict_type == 'Peace Talk':
                text = f"Peace negotiations ongoing in {region}, positive developments"
            elif conflict_type == 'Election':
                text = f"Election-related tensions in {region}, monitoring teams deployed"
            else:
                text = f"{conflict_type} issues reported in {region}"
            
            # Create date
            event_date = start_date + timedelta(
                days=np.random.randint(0, 30),
                hours=np.random.randint(0, 24)
            )
            
            records.append({
                'source': 'synthetic',
                'title': text,
                'description': f"{text}. Authorities monitoring the situation.",
                'content': f"Detailed report: {text}. This incident occurred in {region}. Response teams are on site.",
                'url': f"https://example.com/event/{i}",
                'published_at': event_date.isoformat(),
                'source_name': 'Synthetic Generator',
                'author': 'System',
                'query': 'Kenya',
                'collected_at': datetime.now().isoformat(),
                'region': region,
                'risk_level': details['risk'],
                'conflict_score': details['conflict'],
                'vader_compound': details['sentiment'],
                'vader_negative': abs(details['sentiment']) if details['sentiment'] < 0 else 0,
                'vader_positive': details['sentiment'] if details['sentiment'] > 0 else 0,
                'latitude': np.random.uniform(-4.0, 4.0),
                'longitude': np.random.uniform(33.0, 41.0),
                'word_count': len(text.split()),
                'has_conflict_keywords': 1 if details['conflict'] > 0.3 else 0,
                'text': text
            })
        
        return pd.DataFrame(records)
    
    def collect_all_no_reddit(self) -> pd.DataFrame:
        """Collect from all sources (except Reddit)"""
        logger.info("🚀 Collecting data (No Reddit needed)...")
        
        all_data = []
        
        # 1. Try NewsAPI (if available)
        news_data = self.fetch_newsapi_safe()
        if not news_data.empty:
            all_data.append(news_data)
        
        # 2. Always try GDELT (free)
        gdelt_data = self.fetch_gdelt_free()
        if not gdelt_data.empty:
            all_data.append(gdelt_data)
        
        # 3. Always add synthetic data for consistency
        synth_data = self.generate_smart_synthetic(50)
        all_data.append(synth_data)
        
        # Combine all data
        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            
            # Ensure all required columns exist
            required_columns = ['risk_level', 'conflict_score', 'vader_compound', 
                              'has_conflict_keywords', 'text', 'region', 'source']
            
            for col in required_columns:
                if col not in combined.columns:
                    if col == 'conflict_score':
                        # Generate conflict score based on risk level
                        risk_to_score = {'Low': 0.2, 'Medium': 0.5, 'High': 0.7, 'Critical': 0.9}
                        combined[col] = combined['risk_level'].map(risk_to_score).fillna(0.5)
                    elif col == 'vader_compound':
                        # Generate sentiment based on risk level
                        risk_to_sentiment = {'Low': 0.3, 'Medium': -0.1, 'High': -0.5, 'Critical': -0.8}
                        combined[col] = combined['risk_level'].map(risk_to_sentiment).fillna(0)
                    elif col == 'has_conflict_keywords':
                        combined[col] = (combined['risk_level'].isin(['High', 'Critical'])).astype(int)
                    elif col == 'text':
                        combined[col] = combined.get('title', '') + ' ' + combined.get('description', '')
                    else:
                        combined[col] = ''
            
            # Save to file
            os.makedirs('data/raw', exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"data/raw/collected_no_reddit_{timestamp}.csv"
            combined.to_csv(filename, index=False)
            
            logger.info(f"✅ Collected {len(combined)} total records (No Reddit)")
            logger.info(f"💾 Saved to {filename}")
            
            # Data source breakdown
            source_counts = combined['source'].value_counts()
            for source, count in source_counts.items():
                logger.info(f"   {source}: {count} records")
            
            return combined
        
        return pd.DataFrame()

def main():
    """Test the Reddit-free collector"""
    collector = SimpleDataCollector()
    data = collector.collect_all_no_reddit()
    
    if not data.empty:
        print("\n🎉 SUCCESS! Data collected without Reddit API:")
        print(f"Total records: {len(data)}")
        print("\nColumns available:", list(data.columns))
        print("\nRisk Level distribution:")
        print(data['risk_level'].value_counts())
        
        # Save for dashboard
        data.to_csv("data/processed/latest_no_reddit.csv", index=False)
        print("\n✅ Data saved to 'data/processed/latest_no_reddit.csv'")
    else:
        print("⚠️ No data collected. Check your NewsAPI key or internet connection.")

if __name__ == "__main__":
    main()