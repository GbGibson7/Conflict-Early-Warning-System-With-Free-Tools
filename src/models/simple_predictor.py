"""
Simple ML Predictor - Enhanced with Google Trends
Fixed for categorical data with Google Trends features
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
    """Simple predictor with Google Trends features"""
    
    def __init__(self, model_dir: str = 'models'):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        self.model = None
        self.label_encoder = LabelEncoder()
        self.source_encoder = LabelEncoder()
        
        # Google Trends configuration
        self.google_trends_weight = 0.3  # Weight for Google Trends in predictions
        
        # Initialize with default model
        self.load_or_train_default()
    
    def enhance_with_google_trends(self, text: str, region: str) -> Dict:
        """Enhance prediction with Google Trends data"""
        try:
            # Import inside function to avoid circular imports
            try:
                from src.data_collection.google_trends_collector import GoogleTrendsCollector
                GOOGLE_TRENDS_AVAILABLE = True
            except ImportError:
                GOOGLE_TRENDS_AVAILABLE = False
                logger.info("Google Trends collector not available")
            
            if not GOOGLE_TRENDS_AVAILABLE:
                return {
                    'regional_interest': 0,
                    'overall_trends_risk': 0,
                    'has_trends_data': False,
                    'message': 'Google Trends not installed'
                }
            
            collector = GoogleTrendsCollector()
            
            # Get trends for the region
            regional_data = collector.get_interest_by_region('violence', 'REGION')
            
            if not regional_data.empty:
                # Try to match the region (case-insensitive)
                region_matches = regional_data[
                    regional_data['region'].str.lower().str.contains(region.lower(), na=False)
                ]
                
                if not region_matches.empty:
                    region_score = region_matches['interest_score'].mean()
                else:
                    # If specific region not found, use Kenya average
                    kenya_data = regional_data[regional_data['region'].str.lower() == 'kenya']
                    region_score = kenya_data['interest_score'].mean() if not kenya_data.empty else 0
                
                # Get overall trends
                trends_data = collector.collect_all_trends_data()
                trends_risk = trends_data.get('risk_score', 0)
                
                logger.info(f"Google Trends data for {region}: interest={region_score}, risk={trends_risk}")
                
                return {
                    'regional_interest': float(region_score) if not pd.isna(region_score) else 0,
                    'overall_trends_risk': float(trends_risk),
                    'has_trends_data': True
                }
                
        except Exception as e:
            logger.warning(f"Could not get Google Trends data: {e}")
        
        return {
            'regional_interest': 0,
            'overall_trends_risk': 0,
            'has_trends_data': False
        }
    
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
                # Also try to load label encoder
                encoder_path = os.path.join(self.model_dir, 'label_encoder.joblib')
                if os.path.exists(encoder_path):
                    self.label_encoder = joblib.load(encoder_path)
                logger.info("✅ Loaded existing model")
            except Exception as e:
                logger.warning(f"Could not load model: {e}. Training new one...")
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
        """Predict risk with Google Trends enhancement"""
        if self.model is None:
            self.load_or_train_default()
        
        # Get base prediction
        base_prediction = self._get_base_prediction(text, features)
        
        # Enhance with Google Trends if available
        if features and 'region' in features and features['region']:
            try:
                trends_data = self.enhance_with_google_trends(text, features['region'])
                
                if trends_data['has_trends_data']:
                    # Adjust risk based on Google Trends
                    regional_interest = trends_data['regional_interest']
                    overall_risk = trends_data['overall_trends_risk']
                    
                    # Normalize regional interest (0-100 scale to 0-1)
                    normalized_interest = min(regional_interest / 100, 1.0)
                    
                    # Combine factors (70% regional interest, 30% overall risk)
                    trends_factor = (
                        normalized_interest * 0.7 + 
                        overall_risk * 0.3
                    )
                    
                    # Get current confidence
                    current_confidence = base_prediction.get('confidence', 0.7)
                    
                    # Apply Google Trends weight (blend with original confidence)
                    enhanced_confidence = (
                        current_confidence * (1 - self.google_trends_weight) +
                        trends_factor * self.google_trends_weight
                    )
                    
                    # Adjust risk level if Google Trends suggests higher risk
                    original_risk = base_prediction['risk_level']
                    risk_order = {'Low': 0, 'Medium': 1, 'High': 2, 'Critical': 3}
                    
                    if trends_factor > 0.7 and risk_order.get(original_risk, 1) < 2:
                        # Google Trends suggests high risk, upgrade prediction
                        base_prediction['risk_level'] = 'High'
                    elif trends_factor > 0.5 and risk_order.get(original_risk, 1) < 1:
                        # Google Trends suggests medium risk, upgrade prediction
                        base_prediction['risk_level'] = 'Medium'
                    
                    base_prediction['confidence'] = enhanced_confidence
                    base_prediction['google_trends_impact'] = trends_factor
                    base_prediction['google_trends_regional_interest'] = regional_interest
                    base_prediction['google_trends_overall_risk'] = overall_risk
                    base_prediction['has_google_trends'] = True
                    
                    logger.info(f"Enhanced prediction with Google Trends: factor={trends_factor:.2f}")
                    
            except Exception as e:
                logger.warning(f"Error enhancing with Google Trends: {e}")
                # Continue with base prediction
        
        return base_prediction
    
    def _get_base_prediction(self, text: str, features: Dict = None) -> Dict:
        """Get base ML prediction without Google Trends"""
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
                'timestamp': datetime.now().isoformat(),
                'has_google_trends': False
            }
            
        except Exception as e:
            logger.error(f"ML prediction error: {e}")
            
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
            'timestamp': datetime.now().isoformat(),
            'has_google_trends': False
        }
    
    def get_recommended_action(self, risk_level: str) -> str:
        """Get recommended action based on risk level"""
        actions = {
            'Critical': '🚨 IMMEDIATE RESPONSE: Activate emergency protocols, deploy rapid response teams, notify all authorities',
            'High': '⚠️ HIGH ALERT: Increase surveillance to maximum, prepare emergency teams, notify regional authorities',
            'Medium': '🔶 ELEVATED AWARENESS: Maintain heightened vigilance, gather intelligence, prepare contingency plans',
            'Low': '✅ NORMAL MONITORING: Continue routine surveillance, standard reporting, regular communication'
        }
        return actions.get(risk_level, 'Monitor situation and gather more information')
    
    def batch_predict(self, texts: list, features_list: list = None) -> list:
        """Predict risk for multiple texts"""
        predictions = []
        
        if features_list is None:
            features_list = [{} for _ in texts]
        
        for text, features in zip(texts, features_list):
            predictions.append(self.predict_risk(text, features))
        
        return predictions
    
    def evaluate_google_trends_impact(self, predictions: list) -> Dict:
        """Evaluate how Google Trends affected predictions"""
        total_predictions = len(predictions)
        if total_predictions == 0:
            return {}
        
        google_trends_used = sum(1 for p in predictions if p.get('has_google_trends', False))
        avg_impact = np.mean([p.get('google_trends_impact', 0) for p in predictions 
                            if p.get('has_google_trends', False)] or [0])
        
        # Count risk level changes
        original_risks = []
        final_risks = []
        
        for pred in predictions:
            if 'original_risk' in pred and 'risk_level' in pred:
                original_risks.append(pred['original_risk'])
                final_risks.append(pred['risk_level'])
        
        changes = sum(1 for o, f in zip(original_risks, final_risks) if o != f)
        
        return {
            'total_predictions': total_predictions,
            'google_trends_used': google_trends_used,
            'google_trends_percentage': (google_trends_used / total_predictions * 100) if total_predictions > 0 else 0,
            'average_impact': avg_impact,
            'risk_level_changes': changes
        }

def test_enhanced_predictor():
    """Test the enhanced predictor with Google Trends"""
    print("🧪 Testing Enhanced Predictor with Google Trends")
    print("=" * 60)
    
    predictor = SimpleConflictPredictor()
    
    test_cases = [
        ("Violent clashes in Nairobi with multiple injuries reported", {'region': 'Nairobi'}),
        ("Peaceful community meeting in Kisumu", {'region': 'Kisumu'}),
        ("Explosion reported in Mombasa port area", {'region': 'Mombasa'}),
        ("Normal day with regular activities in Nakuru", {'region': 'Nakuru'})
    ]
    
    for i, (text, features) in enumerate(test_cases, 1):
        print(f"\nTest {i}:")
        print(f"Text: {text}")
        print(f"Region: {features.get('region', 'Unknown')}")
        
        result = predictor.predict_risk(text, features)
        
        print(f"  Risk Level: {result['risk_level']}")
        print(f"  Confidence: {result['confidence']:.2%}")
        print(f"  Action: {result['recommended_action'][:50]}...")
        
        if result.get('has_google_trends'):
            print(f"  Google Trends Impact: {result.get('google_trends_impact', 0):.2%}")
            print(f"  Regional Interest: {result.get('google_trends_regional_interest', 0):.1f}")
        else:
            print(f"  Google Trends: Not used (may not be installed or region not found)")
    
    # Test batch prediction
    print("\n📊 Testing Batch Prediction:")
    texts = ["Protest in Nairobi", "Peace in Mombasa", "Attack in Kisumu"]
    features_list = [{'region': 'Nairobi'}, {'region': 'Mombasa'}, {'region': 'Kisumu'}]
    
    batch_results = predictor.batch_predict(texts, features_list)
    
    for i, result in enumerate(batch_results, 1):
        print(f"  {i}. {texts[i-1]} → {result['risk_level']} ({result['confidence']:.0%})")
    
    # Evaluate Google Trends impact
    print("\n📈 Google Trends Impact Evaluation:")
    impact_stats = predictor.evaluate_google_trends_impact(batch_results)
    print(f"  Total predictions: {impact_stats.get('total_predictions', 0)}")
    print(f"  Google Trends used: {impact_stats.get('google_trends_used', 0)}")
    print(f"  Usage percentage: {impact_stats.get('google_trends_percentage', 0):.1f}%")
    print(f"  Average impact: {impact_stats.get('average_impact', 0):.2%}")
    
    print("\n" + "=" * 60)
    print("✅ Enhanced predictor test complete!")
    print("\n💡 Note: Google Trends requires internet connection and may have rate limits.")
    print("   Install with: pip install pytrends")

if __name__ == "__main__":
    test_enhanced_predictor()