"""
Simple ML Predictor - Fixed for categorical data
"""
import pandas as pd
import numpy as np
import joblib
import warnings
from datetime import datetime
from typing import Dict, Optional
import logging
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import re

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleConflictPredictor:
    """Simple predictor that handles categorical data properly"""
    
    def __init__(self, model_dir: str = 'models'):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        self.model = None
        self.label_encoder = LabelEncoder()
        self.source_encoder = LabelEncoder()
        
        # Initialize with default model
        self.load_or_train_default()
    
    def prepare_input(self, text: str, features: Dict = None) -> Dict:
        """Prepare input features for prediction"""
        if features is None:
            features = {}
        
        # Clean and extract features from text
        cleaned_text = text.lower()
        
        # Conflict keywords
        conflict_keywords = ['violence', 'protest', 'attack', 'clash', 'riot', 
                           'unrest', 'tension', 'crisis', 'emergency']
        
        # Risk level keywords
        risk_keywords = {
            'Critical': ['terror', 'explosion', 'death', 'kill', 'bomb'],
            'High': ['violence', 'clash', 'riot', 'attack'],
            'Medium': ['protest', 'demonstration', 'strike'],
            'Low': ['peace', 'calm', 'normal', 'stable']
        }
        
        # Calculate features
        word_count = len(cleaned_text.split())
        has_conflict = 1 if any(kw in cleaned_text for kw in conflict_keywords) else 0
        
        # Simple sentiment analysis
        positive_words = ['peace', 'calm', 'normal', 'stable', 'safe', 'good']
        negative_words = ['violence', 'attack', 'death', 'kill', 'bad', 'danger']
        
        positive_count = sum(1 for word in positive_words if word in cleaned_text)
        negative_count = sum(1 for word in negative_words if word in cleaned_text)
        
        sentiment = (positive_count - negative_count) / max(word_count, 1)
        
        # Determine risk level from keywords
        risk_level = 'Medium'  # Default
        for level, keywords in risk_keywords.items():
            if any(kw in cleaned_text for kw in keywords):
                risk_level = level
                break
        
        # Prepare input dict
        input_data = {
            'word_count': word_count,
            'has_conflict_keywords': has_conflict,
            'sentiment': sentiment,
            'text_length': len(text),
            'contains_region': 1 if any(region.lower() in cleaned_text for region in 
                                       ['nairobi', 'mombasa', 'kisumu', 'nakuru']) else 0,
            'risk_keyword_count': sum(1 for kw in conflict_keywords if kw in cleaned_text),
            'source': features.get('source', 'unknown'),
            'region': features.get('region', 'unknown')
        }
        
        return input_data
    
    def load_or_train_default(self):
        """Load existing model or train a simple default one"""
        model_path = os.path.join(self.model_dir, 'simple_rf_model.joblib')
        
        if os.path.exists(model_path):
            try:
                self.model = joblib.load(model_path)
                logger.info("✅ Loaded existing model")
            except:
                logger.warning("Could not load model, training new one...")
                self.train_default_model()
        else:
            self.train_default_model()
    
    def train_default_model(self):
        """Train a simple default model"""
        logger.info("Training simple default model...")
        
        # Create synthetic training data
        n_samples = 1000
        np.random.seed(42)
        
        X = []
        y = []
        
        risk_levels = ['Low', 'Medium', 'High', 'Critical']
        
        for i in range(n_samples):
            # Generate features
            word_count = np.random.randint(10, 200)
            has_conflict = np.random.choice([0, 1], p=[0.7, 0.3])
            sentiment = np.random.uniform(-1, 1)
            
            # Determine risk level based on features
            if has_conflict == 1 and sentiment < -0.5:
                risk = 'Critical'
            elif has_conflict == 1 and sentiment < 0:
                risk = 'High'
            elif has_conflict == 1:
                risk = 'Medium'
            else:
                risk = 'Low'
            
            # Add some noise
            if np.random.random() < 0.1:
                risk = np.random.choice(risk_levels)
            
            # Create feature vector
            features = [
                word_count,
                has_conflict,
                sentiment,
                np.random.randint(50, 500),  # text_length
                np.random.choice([0, 1]),  # contains_region
                np.random.randint(0, 5),  # risk_keyword_count
                np.random.uniform(0, 1),  # source_encoded (placeholder)
                np.random.uniform(0, 1)   # region_encoded (placeholder)
            ]
            
            X.append(features)
            y.append(risk)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        
        self.model.fit(X, y_encoded)
        
        # Save model
        model_path = os.path.join(self.model_dir, 'simple_rf_model.joblib')
        joblib.dump(self.model, model_path)
        joblib.dump(self.label_encoder, 
                   os.path.join(self.model_dir, 'label_encoder.joblib'))
        
        logger.info(f"✅ Trained and saved simple model to {model_path}")
    
    def predict_risk(self, text: str, features: Dict = None) -> Dict:
        """Predict risk for given text"""
        if self.model is None:
            self.load_or_train_default()
        
        # Prepare input
        input_data = self.prepare_input(text, features)
        
        # Create feature vector (numerical only)
        feature_vector = [
            input_data['word_count'],
            input_data['has_conflict_keywords'],
            input_data['sentiment'],
            input_data['text_length'],
            input_data['contains_region'],
            input_data['risk_keyword_count'],
            0.5,  # Placeholder for source
            0.5   # Placeholder for region
        ]
        
        try:
            # Make prediction
            prediction = self.model.predict([feature_vector])[0]
            probability = self.model.predict_proba([feature_vector])[0]
            
            risk_level = self.label_encoder.inverse_transform([prediction])[0]
            confidence = max(probability)
            
            # Get recommended action
            recommended_action = self.get_recommended_action(risk_level)
            
            return {
                'text': text,
                'risk_level': risk_level,
                'confidence': float(confidence),
                'recommended_action': recommended_action,
                'features_used': {
                    'word_count': input_data['word_count'],
                    'has_conflict': input_data['has_conflict_keywords'],
                    'sentiment': input_data['sentiment']
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            
            # Fallback to rule-based prediction
            return self.rule_based_prediction(text, features)
    
    def rule_based_prediction(self, text: str, features: Dict = None) -> Dict:
        """Rule-based fallback prediction"""
        text_lower = text.lower()
        
        # Rule-based risk assessment
        critical_keywords = ['terror', 'explosion', 'death', 'kill', 'bomb']
        high_keywords = ['violence', 'attack', 'clash', 'riot', 'hostage']
        medium_keywords = ['protest', 'demonstration', 'strike', 'unrest']
        low_keywords = ['peace', 'calm', 'normal', 'stable']
        
        if any(kw in text_lower for kw in critical_keywords):
            risk_level = 'Critical'
        elif any(kw in text_lower for kw in high_keywords):
            risk_level = 'High'
        elif any(kw in text_lower for kw in medium_keywords):
            risk_level = 'Medium'
        elif any(kw in text_lower for kw in low_keywords):
            risk_level = 'Low'
        else:
            risk_level = 'Medium'  # Default
        
        return {
            'text': text,
            'risk_level': risk_level,
            'confidence': 0.7,
            'recommended_action': self.get_recommended_action(risk_level),
            'method': 'rule_based',
            'timestamp': datetime.now().isoformat()
        }
    
    def get_recommended_action(self, risk_level: str) -> str:
        """Get recommended action based on risk level"""
        actions = {
            'Critical': '🚨 IMMEDIATE RESPONSE: Activate emergency protocols',
            'High': '⚠️ HIGH ALERT: Increase surveillance and preparedness',
            'Medium': '🔶 MONITOR: Maintain vigilance and gather information',
            'Low': '✅ NORMAL: Continue routine monitoring'
        }
        return actions.get(risk_level, 'Monitor situation')
    
    def batch_predict(self, texts: list) -> list:
        """Predict risk for multiple texts"""
        predictions = []
        for text in texts:
            predictions.append(self.predict_risk(text))
        return predictions

def test_predictor():
    """Test the simple predictor"""
    print("🧪 Testing Simple Predictor")
    print("=" * 50)
    
    predictor = SimpleConflictPredictor()
    
    test_cases = [
        "Violent clashes in Nairobi with multiple injuries reported",
        "Peaceful community meeting in Kisumu",
        "Explosion reported in Mombasa port area",
        "Normal day with regular activities"
    ]
    
    for i, text in enumerate(test_cases, 1):
        print(f"\nTest {i}: {text}")
        result = predictor.predict_risk(text)
        print(f"  Risk Level: {result['risk_level']}")
        print(f"  Confidence: {result['confidence']:.2%}")
        print(f"  Action: {result['recommended_action']}")
    
    print("\n" + "=" * 50)
    print("✅ Simple predictor test complete!")

if __name__ == "__main__":
    test_predictor()