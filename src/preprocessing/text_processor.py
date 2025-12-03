"""
Text Processing for Conflict Analysis
Uses free NLP libraries
"""
import re
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from datetime import datetime

# NLP Libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import spacy

# Download NLTK data
try:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
except:
    pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextProcessor:
    """Process text for conflict analysis"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.vader = SentimentIntensityAnalyzer()
        self.nlp = spacy.load('en_core_web_sm')
        
        # Conflict keywords (DO NOT remove these)
        self.conflict_keywords = {
            'conflict', 'war', 'peace', 'violence', 'attack', 'protest',
            'demonstration', 'riot', 'clash', 'tension', 'security',
            'unrest', 'ceasefire', 'mediation', 'negotiation', 'fight',
            'battle', 'terror', 'attack', 'kill', 'death', 'injured',
            'crisis', 'emergency', 'evacuate', 'hostage', 'bomb',
            'explosion', 'shooting', 'arrest', 'detention', 'missing'
        }
        
        # Remove conflict keywords from stopwords
        self.stop_words = self.stop_words - self.conflict_keywords
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not isinstance(text, str):
            return ""
            
        text = str(text).lower().strip()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def extract_entities(self, text: str) -> Dict:
        """Extract named entities using spaCy"""
        doc = self.nlp(text)
        
        entities = {
            'persons': [],
            'organizations': [],
            'locations': [],
            'dates': [],
            'money': [],
            'gpe': []  # Countries, cities, states
        }
        
        for ent in doc.ents:
            if ent.label_ == 'PERSON':
                entities['persons'].append(ent.text)
            elif ent.label_ == 'ORG':
                entities['organizations'].append(ent.text)
            elif ent.label_ == 'LOC':
                entities['locations'].append(ent.text)
            elif ent.label_ == 'DATE':
                entities['dates'].append(ent.text)
            elif ent.label_ == 'MONEY':
                entities['money'].append(ent.text)
            elif ent.label_ == 'GPE':
                entities['gpe'].append(ent.text)
                
        return entities
    
    def analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment using multiple methods"""
        cleaned_text = self.clean_text(text)
        
        if not cleaned_text or len(cleaned_text.split()) < 3:
            return self._get_empty_sentiment()
        
        # VADER Sentiment (good for social media)
        vader_scores = self.vader.polarity_scores(cleaned_text)
        
        # TextBlob Sentiment
        blob = TextBlob(cleaned_text)
        textblob_polarity = blob.sentiment.polarity
        textblob_subjectivity = blob.sentiment.subjectivity
        
        # Conflict detection
        conflict_score = self._calculate_conflict_score(cleaned_text)
        
        # Risk assessment
        risk_level = self._assess_risk_level(
            vader_scores['compound'],
            conflict_score,
            textblob_subjectivity
        )
        
        # Entities
        entities = self.extract_entities(text)
        
        return {
            'text': text,
            'cleaned_text': cleaned_text,
            'vader_compound': round(vader_scores['compound'], 3),
            'vader_positive': round(vader_scores['pos'], 3),
            'vader_negative': round(vader_scores['neg'], 3),
            'vader_neutral': round(vader_scores['neu'], 3),
            'textblob_polarity': round(textblob_polarity, 3),
            'textblob_subjectivity': round(textblob_subjectivity, 3),
            'conflict_score': round(conflict_score, 3),
            'risk_level': risk_level,
            'word_count': len(cleaned_text.split()),
            'has_conflict_keywords': conflict_score > 0.3,
            'entities': entities,
            'sentiment_label': self._get_sentiment_label(vader_scores['compound']),
            'processed_at': datetime.now().isoformat()
        }
    
    def _calculate_conflict_score(self, text: str) -> float:
        """Calculate conflict intensity score"""
        words = text.lower().split()
        
        # Count conflict keywords
        conflict_count = sum(1 for word in words if word in self.conflict_keywords)
        
        # Check for intensity indicators
        intensity_indicators = [
            'very', 'extremely', 'highly', 'severely', 'critically',
            'massive', 'large', 'major', 'serious'
        ]
        
        intensity_count = sum(1 for word in words if word in intensity_indicators)
        
        # Calculate score (0 to 1)
        base_score = min(conflict_count / 5, 1.0)
        intensity_boost = intensity_count * 0.1
        
        return min(base_score + intensity_boost, 1.0)
    
    def _assess_risk_level(self, sentiment: float, conflict_score: float, 
                          subjectivity: float) -> str:
        """Assess overall risk level"""
        # Negative sentiment + high conflict + subjective = high risk
        risk_score = (
            abs(sentiment) * 0.3 +  # Negative sentiment contributes
            conflict_score * 0.5 +   # Conflict keywords are important
            subjectivity * 0.2       # Subjective language can indicate bias
        )
        
        if risk_score > 0.7:
            return 'Critical'
        elif risk_score > 0.5:
            return 'High'
        elif risk_score > 0.3:
            return 'Medium'
        else:
            return 'Low'
    
    def _get_sentiment_label(self, compound_score: float) -> str:
        """Convert sentiment score to label"""
        if compound_score >= 0.05:
            return 'Positive'
        elif compound_score <= -0.05:
            return 'Negative'
        else:
            return 'Neutral'
    
    def _get_empty_sentiment(self) -> Dict:
        """Return empty sentiment analysis"""
        return {
            'text': '',
            'cleaned_text': '',
            'vader_compound': 0,
            'vader_positive': 0,
            'vader_negative': 0,
            'vader_neutral': 1,
            'textblob_polarity': 0,
            'textblob_subjectivity': 0,
            'conflict_score': 0,
            'risk_level': 'Low',
            'word_count': 0,
            'has_conflict_keywords': False,
            'entities': {},
            'sentiment_label': 'Neutral',
            'processed_at': datetime.now().isoformat()
        }
    
    def process_dataframe(self, df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
        """Process all texts in a DataFrame"""
        if text_column not in df.columns:
            logger.error(f"Column '{text_column}' not found")
            return df
            
        logger.info(f"Processing {len(df)} texts...")
        
        results = []
        for idx, row in df.iterrows():
            if idx % 50 == 0 and idx > 0:
                logger.info(f"  Processed {idx}/{len(df)}")
                
            text = row.get(text_column, '')
            analysis = self.analyze_sentiment(text)
            
            # Combine with original row data
            result = {**row.to_dict(), **analysis}
            results.append(result)
            
        processed_df = pd.DataFrame(results)
        
        logger.info(f"✅ Processed {len(processed_df)} records")
        return processed_df

def test_processor():
    """Test the text processor"""
    print("🧪 Testing Text Processor")
    print("=" * 50)
    
    processor = TextProcessor()
    
    test_cases = [
        "Violent protests in Nairobi today with multiple injuries reported.",
        "Peaceful negotiations continue between government and opposition.",
        "Explosion heard in Mombasa port area, emergency services responding.",
        "Normal day with regular activities across the country."
    ]
    
    for i, text in enumerate(test_cases, 1):
        print(f"\nTest {i}:")
        print(f"Text: {text}")
        
        analysis = processor.analyze_sentiment(text)
        
        print(f"  Sentiment: {analysis['sentiment_label']}")
        print(f"  Risk Level: {analysis['risk_level']}")
        print(f"  Conflict Score: {analysis['conflict_score']}")
        print(f"  VADER Score: {analysis['vader_compound']}")
        
        if analysis['entities']['locations']:
            print(f"  Locations: {analysis['entities']['locations']}")
    
    print("\n" + "=" * 50)
    print("✅ Text processing test complete!")

if __name__ == "__main__":
    test_processor()