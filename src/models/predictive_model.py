"""
Machine Learning for Conflict Prediction
100% Free using scikit-learn
"""
import pandas as pd
import numpy as np
import joblib
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
import os

# ML Libraries
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, classification_report
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Imbalanced learning
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import make_pipeline as make_imbalance_pipeline

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConflictPredictor:
    """Predict conflict risk using ML"""
    
    def __init__(self, model_dir: str = 'models'):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        # Models
        self.models = {}
        self.best_model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.vectorizer = TfidfVectorizer(max_features=1000)
        
        # Feature configuration
        self.numeric_features = [
            'vader_compound', 'vader_negative', 'conflict_score',
            'word_count', 'has_conflict_keywords'
        ]
        
        self.categorical_features = ['source', 'region']
        self.text_features = ['cleaned_text']
        
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """Prepare features for ML"""
        logger.info("Preparing features...")
        
        # Ensure required columns exist
        for col in self.numeric_features:
            if col not in df.columns:
                df[col] = 0
        
        # Handle categorical features
        for col in self.categorical_features:
            if col in df.columns:
                df[col] = df[col].fillna('unknown').astype(str)
            else:
                df[col] = 'unknown'
        
        # Prepare text features
        if 'cleaned_text' not in df.columns:
            df['cleaned_text'] = df.get('text', '').fillna('')
        
        # Prepare target variable
        if 'risk_level' not in df.columns:
            # Create synthetic target based on features
            df['risk_level'] = self._create_synthetic_target(df)
        
        # Encode target
        y = self.label_encoder.fit_transform(df['risk_level'])
        
        return df, y
    
    def _create_synthetic_target(self, df: pd.DataFrame) -> pd.Series:
        """Create synthetic target variable for training"""
        # Simple rule-based target creation
        conditions = [
            (df['conflict_score'] > 0.7) | (df['vader_negative'] > 0.5),
            (df['conflict_score'] > 0.5) | (df['vader_negative'] > 0.3),
            (df['conflict_score'] > 0.3),
            True
        ]
        
        choices = ['Critical', 'High', 'Medium', 'Low']
        
        return pd.Series(
            np.select(conditions, choices, default='Low'),
            index=df.index
        )
    
    def create_training_data(self, n_samples: int = 2000) -> pd.DataFrame:
        """Create comprehensive training data"""
        logger.info(f"Creating {n_samples} training samples...")
        
        np.random.seed(42)
        
        data = []
        
        # Event types and their characteristics
        event_types = {
            'Peaceful': {'conflict_range': (0.0, 0.2), 'sentiment_range': (0.1, 0.8)},
            'Tension': {'conflict_range': (0.2, 0.4), 'sentiment_range': (-0.3, 0.1)},
            'Protest': {'conflict_range': (0.4, 0.6), 'sentiment_range': (-0.6, -0.1)},
            'Violence': {'conflict_range': (0.6, 0.8), 'sentiment_range': (-0.9, -0.4)},
            'Crisis': {'conflict_range': (0.8, 1.0), 'sentiment_range': (-1.0, -0.7)}
        }
        
        regions = ['Nairobi', 'Mombasa', 'Kisumu', 'Nakuru', 'Eldoret']
        sources = ['newsapi', 'reddit', 'gdelt', 'synthetic']
        
        for i in range(n_samples):
            # Randomly select event type
            event_type = np.random.choice(list(event_types.keys()))
            char = event_types[event_type]
            
            # Generate features
            conflict_score = np.random.uniform(*char['conflict_range'])
            sentiment = np.random.uniform(*char['sentiment_range'])
            word_count = np.random.randint(20, 500)
            
            # Create text based on event type
            region = np.random.choice(regions)
            if event_type == 'Peaceful':
                text = f"Peaceful activities in {region}. Normal day."
                risk_level = 'Low'
            elif event_type == 'Tension':
                text = f"Growing tension in {region}. Discussions ongoing."
                risk_level = 'Medium'
            elif event_type == 'Protest':
                text = f"Protests reported in {region}. Police monitoring."
                risk_level = 'High'
            elif event_type == 'Violence':
                text = f"Violent clashes in {region}. Casualties reported."
                risk_level = 'High'
            else:  # Crisis
                text = f"Critical situation in {region}. Emergency declared."
                risk_level = 'Critical'
            
            # Create record
            record = {
                'source': np.random.choice(sources),
                'region': region,
                'text': text,
                'cleaned_text': text.lower(),
                'vader_compound': sentiment,
                'vader_negative': max(-sentiment, 0) if sentiment < 0 else 0,
                'conflict_score': conflict_score,
                'word_count': word_count,
                'has_conflict_keywords': int(conflict_score > 0.3),
                'risk_level': risk_level,
                'created_at': datetime.now() - timedelta(days=np.random.randint(0, 365)),
                'latitude': np.random.uniform(-4.0, 4.0),
                'longitude': np.random.uniform(33.0, 41.0)
            }
            
            data.append(record)
        
        return pd.DataFrame(data)
    
    def build_model_pipeline(self):
        """Build ML pipeline"""
        logger.info("Building ML pipeline...")
        
        # Define preprocessing
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('onehot', 'passthrough')  # Simplified for demo
        ])
        
        text_transformer = Pipeline(steps=[
            ('tfidf', TfidfVectorizer(max_features=500, stop_words='english'))
        ])
        
        # Column transformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features),
                ('text', text_transformer, 'cleaned_text')
            ])
        
        # Define models
        model_configs = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            ),
            'logistic_regression': LogisticRegression(
                random_state=42,
                max_iter=1000,
                class_weight='balanced',
                solver='liblinear'
            ),
            'svm': SVC(
                probability=True,
                random_state=42,
                class_weight='balanced',
                kernel='rbf'
            ),
            'neural_network': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=1000,
                random_state=42,
                early_stopping=True
            )
        }
        
        # Create pipelines with SMOTE for handling imbalance
        self.pipelines = {}
        for name, model in model_configs.items():
            self.pipelines[name] = make_imbalance_pipeline(
                preprocessor,
                SMOTE(random_state=42),
                model
            )
            
        logger.info(f"Built {len(self.pipelines)} model pipelines")
        return self.pipelines
    
    def train_models(self, X_train: pd.DataFrame, y_train: np.ndarray,
                    X_test: pd.DataFrame = None, y_test: np.ndarray = None) -> Dict:
        """Train all models"""
        logger.info("Training models...")
        
        if not hasattr(self, 'pipelines'):
            self.build_model_pipeline()
        
        results = {}
        
        for name, pipeline in self.pipelines.items():
            logger.info(f"  Training {name}...")
            
            # Train the model
            pipeline.fit(X_train, y_train)
            
            # Store the model
            self.models[name] = pipeline
            
            # Evaluate on training data
            y_train_pred = pipeline.predict(X_train)
            train_accuracy = accuracy_score(y_train, y_train_pred)
            
            # If test data provided, evaluate
            test_metrics = {}
            if X_test is not None and y_test is not None:
                y_test_pred = pipeline.predict(X_test)
                y_test_proba = pipeline.predict_proba(X_test)
                
                test_metrics = {
                    'accuracy': accuracy_score(y_test, y_test_pred),
                    'precision': precision_score(y_test, y_test_pred, average='weighted'),
                    'recall': recall_score(y_test, y_test_pred, average='weighted'),
                    'f1_score': f1_score(y_test, y_test_pred, average='weighted'),
                    'predictions': y_test_pred,
                    'probabilities': y_test_proba
                }
                
                # Cross-validation scores
                cv_scores = cross_val_score(pipeline, X_train, y_train, 
                                          cv=5, scoring='accuracy')
                test_metrics['cv_mean'] = cv_scores.mean()
                test_metrics['cv_std'] = cv_scores.std()
            
            results[name] = {
                'train_accuracy': train_accuracy,
                'test_metrics': test_metrics,
                'model': pipeline
            }
            
            logger.info(f"    Train Accuracy: {train_accuracy:.3f}")
            if test_metrics:
                logger.info(f"    Test Accuracy: {test_metrics['accuracy']:.3f}")
        
        # Select best model
        if results:
            best_model_name = max(
                results.keys(),
                key=lambda x: results[x]['test_metrics'].get('f1_score', 0) 
                if results[x]['test_metrics'] else results[x]['train_accuracy']
            )
            self.best_model = results[best_model_name]['model']
            logger.info(f"✅ Best model: {best_model_name}")
        
        return results
    
    def predict(self, text: str, features: Dict = None) -> Dict:
        """Make prediction for new text"""
        if self.best_model is None:
            logger.warning("No trained model found. Training default model...")
            self.train_default()
        
        # Prepare input data
        if features is None:
            features = {}
        
        # Create a DataFrame with the input
        input_data = {
            'text': text,
            'cleaned_text': text.lower(),
            'vader_compound': features.get('vader_compound', 0),
            'vader_negative': features.get('vader_negative', 0),
            'conflict_score': features.get('conflict_score', 0),
            'word_count': len(text.split()),
            'has_conflict_keywords': features.get('has_conflict_keywords', 0),
            'source': features.get('source', 'unknown'),
            'region': features.get('region', 'unknown')
        }
        
        input_df = pd.DataFrame([input_data])
        
        # Make predictions with all models
        predictions = {}
        for name, model in self.models.items():
            try:
                y_pred = model.predict(input_df)[0]
                y_proba = model.predict_proba(input_df)[0]
                
                predictions[name] = {
                    'risk_level': self.label_encoder.inverse_transform([y_pred])[0],
                    'confidence': float(max(y_proba)),
                    'probabilities': y_proba.tolist()
                }
            except Exception as e:
                logger.error(f"Error in {name}: {e}")
                predictions[name] = {
                    'risk_level': 'Unknown',
                    'confidence': 0.0,
                    'probabilities': []
                }
        
        # Ensemble prediction (weighted by confidence)
        risk_scores = {}
        for name, pred in predictions.items():
            risk_level = pred['risk_level']
            confidence = pred['confidence']
            
            if risk_level not in risk_scores:
                risk_scores[risk_level] = 0
            risk_scores[risk_level] += confidence
        
        if risk_scores:
            ensemble_risk = max(risk_scores, key=risk_scores.get)
        else:
            ensemble_risk = 'Medium'
        
        return {
            'text': text,
            'ensemble_risk': ensemble_risk,
            'model_predictions': predictions,
            'recommended_action': self.get_recommended_action(ensemble_risk),
            'timestamp': datetime.now().isoformat()
        }
    
    def get_recommended_action(self, risk_level: str) -> str:
        """Get recommended action based on risk level"""
        actions = {
            'Critical': '🚨 IMMEDIATE RESPONSE: Activate emergency protocols, deploy rapid response teams, notify all authorities, prepare evacuation if needed',
            'High': '⚠️ HIGH ALERT: Increase surveillance to maximum, prepare emergency teams on standby, notify regional authorities, monitor 24/7',
            'Medium': '🔶 ELEVATED AWARENESS: Maintain heightened vigilance, gather additional intelligence, prepare contingency plans, regular status updates',
            'Low': '✅ NORMAL MONITORING: Continue routine surveillance, standard reporting procedures, maintain regular communication channels'
        }
        return actions.get(risk_level, 'Monitor situation and gather more information')
    
    def train_default(self, n_samples: int = 1000):
        """Train default models with synthetic data"""
        logger.info("Training default models with synthetic data...")
        
        # Generate synthetic data
        train_df = self.create_training_data(n_samples)
        
        # Prepare features
        X = train_df
        y = self.label_encoder.fit_transform(train_df['risk_level'])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train models
        results = self.train_models(X_train, y_train, X_test, y_test)
        
        # Save models
        self.save_models()
        
        return results
    
    def save_models(self):
        """Save trained models to disk"""
        logger.info(f"Saving models to {self.model_dir}...")
        
        # Save each model
        for name, model in self.models.items():
            filename = os.path.join(self.model_dir, f'{name}.joblib')
            joblib.dump(model, filename)
            logger.info(f"  Saved {name} to {filename}")
        
        # Save other components
        joblib.dump(self.label_encoder, 
                   os.path.join(self.model_dir, 'label_encoder.joblib'))
        
        logger.info("✅ All models saved successfully")
    
    def load_models(self):
        """Load models from disk"""
        logger.info(f"Loading models from {self.model_dir}...")
        
        model_files = [
            'random_forest.joblib',
            'gradient_boosting.joblib',
            'logistic_regression.joblib'
        ]
        
        for model_file in model_files:
            filepath = os.path.join(self.model_dir, model_file)
            if os.path.exists(filepath):
                name = model_file.replace('.joblib', '')
                self.models[name] = joblib.load(filepath)
                logger.info(f"  Loaded {name}")
        
        # Load label encoder
        encoder_path = os.path.join(self.model_dir, 'label_encoder.joblib')
        if os.path.exists(encoder_path):
            self.label_encoder = joblib.load(encoder_path)
        
        # Set best model (use random forest as default)
        if 'random_forest' in self.models:
            self.best_model = self.models['random_forest']
        
        logger.info(f"✅ Loaded {len(self.models)} models")
    
    def evaluate_model(self, model_name: str, X_test: pd.DataFrame, 
                      y_test: np.ndarray) -> Dict:
        """Evaluate a specific model"""
        if model_name not in self.models:
            logger.error(f"Model {model_name} not found")
            return {}
        
        model = self.models[model_name]
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision_weighted': precision_score(y_test, y_pred, average='weighted'),
            'recall_weighted': recall_score(y_test, y_pred, average='weighted'),
            'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'risk_levels': self.label_encoder.classes_.tolist()
        }
        
        return metrics
    
    def analyze_trends(self, df: pd.DataFrame, time_column: str = 'created_at') -> Dict:
        """Analyze temporal trends in conflict data"""
        if time_column not in df.columns:
            logger.error(f"Time column {time_column} not found")
            return {}
        
        # Convert to datetime
        df[time_column] = pd.to_datetime(df[time_column])
        
        # Group by time periods
        df['date'] = df[time_column].dt.date
        df['hour'] = df[time_column].dt.hour
        df['day_of_week'] = df[time_column].dt.dayofweek
        
        # Daily trends
        daily_trends = df.groupby('date').agg({
            'conflict_score': 'mean',
            'vader_compound': 'mean',
            'risk_level': lambda x: (x == 'High').sum() + (x == 'Critical').sum()
        }).reset_index()
        
        # Hourly patterns
        hourly_patterns = df.groupby('hour').agg({
            'conflict_score': 'mean',
            'vader_negative': 'mean'
        }).reset_index()
        
        # Weekly patterns
        weekly_patterns = df.groupby('day_of_week').agg({
            'conflict_score': 'mean',
            'vader_compound': 'mean'
        }).reset_index()
        
        return {
            'daily_trends': daily_trends.to_dict('records'),
            'hourly_patterns': hourly_patterns.to_dict('records'),
            'weekly_patterns': weekly_patterns.to_dict('records'),
            'total_records': len(df),
            'high_risk_count': len(df[df['risk_level'].isin(['High', 'Critical'])]),
            'avg_conflict_score': df['conflict_score'].mean(),
            'avg_sentiment': df['vader_compound'].mean()
        }

def test_predictor():
    """Test the conflict predictor"""
    print("🧪 Testing Conflict Predictor")
    print("=" * 60)
    
    # Initialize predictor
    predictor = ConflictPredictor(model_dir='test_models')
    
    print("\n1. Generating training data...")
    train_data = predictor.create_training_data(500)
    print(f"   Generated {len(train_data)} samples")
    
    print("\n2. Preparing features...")
    X, y = predictor.prepare_features(train_data)
    
    print("\n3. Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print("\n4. Building and training models...")
    results = predictor.train_models(X_train, y_train, X_test, y_test)
    
    print("\n5. Testing predictions...")
    test_texts = [
        "Violent clashes in Nairobi CBD, multiple injuries reported, police deployed",
        "Peaceful community meeting in Kisumu discussing development projects",
        "Protest turning violent near government building, tear gas used",
        "Normal day with regular activities and peaceful demonstrations"
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\nTest {i}:")
        print(f"Text: {text}")
        
        # Create basic features
        features = {
            'vader_compound': -0.8 if 'violent' in text else (0.5 if 'peaceful' in text else 0),
            'conflict_score': 0.9 if 'violent' in text else (0.1 if 'peaceful' in text else 0.3),
            'region': 'Nairobi' if 'Nairobi' in text else 'Kisumu'
        }
        
        prediction = predictor.predict(text, features)
        
        print(f"  Risk Level: {prediction['ensemble_risk']}")
        print(f"  Recommended Action: {prediction['recommended_action']}")
        
        # Show model consensus
        print(f"  Model Consensus:")
        for model_name, pred in prediction['model_predictions'].items():
            print(f"    {model_name}: {pred['risk_level']} ({pred['confidence']:.2f})")
    
    print("\n" + "=" * 60)
    print("✅ ML testing complete!")
    
    # Save models
    predictor.save_models()

if __name__ == "__main__":
    test_predictor()                