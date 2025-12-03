"""
Visualization for Conflict Data
Uses Plotly for interactive charts
"""
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConflictVisualizer:
    """Generate visualizations for conflict data"""
    
    def __init__(self):
        self.color_map = {
            'Critical': '#FF0000',  # Red
            'High': '#FF6B00',      # Orange
            'Medium': '#FFD700',    # Yellow
            'Low': '#00FF00'        # Green
        }
        
        self.sentiment_colors = {
            'Positive': '#1E88E5',  # Blue
            'Neutral': '#FFC107',   # Amber
            'Negative': '#E53935'   # Red
        }
    
    def create_risk_dashboard(self, df: pd.DataFrame) -> Dict:
        """Create complete dashboard with multiple visualizations"""
        dashboard = {}
        
        # 1. Risk Distribution Pie Chart
        dashboard['risk_distribution'] = self.plot_risk_distribution(df)
        
        # 2. Temporal Risk Trend
        if 'date' in df.columns or 'created_at' in df.columns:
            time_col = 'date' if 'date' in df.columns else 'created_at'
            dashboard['temporal_trend'] = self.plot_temporal_trend(df, time_col)
        
        # 3. Geographic Heatmap
        if 'latitude' in df.columns and 'longitude' in df.columns:
            dashboard['heatmap'] = self.plot_geographic_heatmap(df)
        
        # 4. Sentiment Analysis
        if 'vader_compound' in df.columns:
            dashboard['sentiment_analysis'] = self.plot_sentiment_analysis(df)
        
        # 5. Word Cloud Data
        if 'cleaned_text' in df.columns:
            dashboard['word_cloud_data'] = self.generate_word_frequencies(df)
        
        # 6. Summary Statistics
        dashboard['summary_stats'] = self.generate_summary_stats(df)
        
        return dashboard
    
    def plot_risk_distribution(self, df: pd.DataFrame) -> Dict:
        """Create pie chart of risk distribution"""
        if 'risk_level' not in df.columns:
            return {}
        
        risk_counts = df['risk_level'].value_counts().reindex(['Low', 'Medium', 'High', 'Critical'])
        
        fig = go.Figure(data=[
            go.Pie(
                labels=risk_counts.index,
                values=risk_counts.values,
                hole=.3,
                marker_colors=[self.color_map.get(level, '#CCCCCC') for level in risk_counts.index]
            )
        ])
        
        fig.update_layout(
            title_text='Risk Level Distribution',
            showlegend=True,
            height=400
        )
        
        return {
            'plot': fig.to_json(),
            'data': risk_counts.to_dict()
        }
    
    def plot_temporal_trend(self, df: pd.DataFrame, time_column: str = 'date') -> Dict:
        """Plot risk trends over time"""
        df_copy = df.copy()
        df_copy[time_column] = pd.to_datetime(df_copy[time_column])
        df_copy['date_only'] = df_copy[time_column].dt.date
        
        # Group by date and risk level
        daily_risk = df_copy.groupby(['date_only', 'risk_level']).size().unstack(fill_value=0)
        
        fig = go.Figure()
        
        for risk_level in ['Low', 'Medium', 'High', 'Critical']:
            if risk_level in daily_risk.columns:
                fig.add_trace(go.Scatter(
                    x=daily_risk.index,
                    y=daily_risk[risk_level],
                    mode='lines+markers',
                    name=risk_level,
                    line=dict(color=self.color_map.get(risk_level, '#CCCCCC'), width=2),
                    marker=dict(size=8)
                ))
        
        fig.update_layout(
            title_text='Risk Trends Over Time',
            xaxis_title='Date',
            yaxis_title='Number of Events',
            hovermode='x unified',
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        return {
            'plot': fig.to_json(),
            'trend_data': daily_risk.to_dict()
        }
    
    def plot_geographic_heatmap(self, df: pd.DataFrame) -> Dict:
        """Create geographic heatmap of conflict events"""
        # Filter out invalid coordinates
        valid_coords = df.dropna(subset=['latitude', 'longitude'])
        valid_coords = valid_coords[
            (valid_coords['latitude'].between(-90, 90)) & 
            (valid_coords['longitude'].between(-180, 180))
        ]
        
        if valid_coords.empty:
            return {}
        
        # Assign size based on risk level
        size_map = {'Low': 5, 'Medium': 10, 'High': 15, 'Critical': 20}
        valid_coords['marker_size'] = valid_coords['risk_level'].map(size_map).fillna(5)
        
        fig = go.Figure()
        
        # Add scatter map
        fig.add_trace(go.Scattermapbox(
            lat=valid_coords['latitude'],
            lon=valid_coords['longitude'],
            mode='markers',
            marker=dict(
                size=valid_coords['marker_size'],
                color=valid_coords['risk_level'].map(self.color_map),
                opacity=0.7,
                sizemode='diameter'
            ),
            text=valid_coords.apply(
                lambda row: f"{row.get('region', 'Unknown')}: {row.get('risk_level', 'Unknown')}",
                axis=1
            ),
            hoverinfo='text'
        ))
        
        fig.update_layout(
            mapbox_style="open-street-map",
            mapbox=dict(
                center=dict(
                    lat=valid_coords['latitude'].mean(),
                    lon=valid_coords['longitude'].mean()
                ),
                zoom=5
            ),
            margin={"r":0,"t":30,"l":0,"b":0},
            height=500,
            title_text='Geographic Distribution of Conflict Events'
        )
        
        return {
            'plot': fig.to_json(),
            'locations': valid_coords[['latitude', 'longitude', 'risk_level', 'region']].to_dict('records')
        }
    
    def plot_sentiment_analysis(self, df: pd.DataFrame) -> Dict:
        """Create sentiment analysis visualizations"""
        if 'vader_compound' not in df.columns:
            return {}
        
        # Create sentiment distribution histogram
        fig = px.histogram(
            df, 
            x='vader_compound',
            nbins=30,
            title='Sentiment Distribution',
            labels={'vader_compound': 'Sentiment Score (-1 to 1)'},
            color_discrete_sequence=['#1E88E5']
        )
        
        fig.update_layout(
            height=400,
            showlegend=False
        )
        
        # Add vertical lines for sentiment thresholds
        fig.add_vline(x=-0.05, line_dash="dash", line_color="red", 
                     annotation_text="Negative", annotation_position="top")
        fig.add_vline(x=0.05, line_dash="dash", line_color="green", 
                     annotation_text="Positive", annotation_position="top")
        
        # Calculate sentiment statistics
        sentiment_stats = {
            'mean': float(df['vader_compound'].mean()),
            'median': float(df['vader_compound'].median()),
            'std': float(df['vader_compound'].std()),
            'positive_percentage': float((df['vader_compound'] > 0.05).mean() * 100),
            'negative_percentage': float((df['vader_compound'] < -0.05).mean() * 100),
            'neutral_percentage': float(((df['vader_compound'] >= -0.05) & (df['vader_compound'] <= 0.05)).mean() * 100)
        }
        
        return {
            'plot': fig.to_json(),
            'statistics': sentiment_stats
        }
    
    def generate_word_frequencies(self, df: pd.DataFrame, top_n: int = 20) -> Dict:
        """Generate word frequencies for word cloud"""
        if 'cleaned_text' not in df.columns:
            return {}
        
        # Combine all text
        all_text = ' '.join(df['cleaned_text'].dropna().astype(str))
        
        # Split into words and count frequencies
        words = all_text.lower().split()
        word_freq = {}
        
        # Common words to exclude
        stop_words = {
            'the', 'and', 'in', 'of', 'to', 'a', 'is', 'for', 'on', 'that',
            'with', 'by', 'this', 'are', 'as', 'it', 'be', 'was', 'were',
            'has', 'have', 'had', 'but', 'not', 'at', 'from', 'or', 'an'
        }
        
        for word in words:
            if word not in stop_words and len(word) > 3:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top N words
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        return {
            'word_frequencies': dict(top_words),
            'total_words': len(words),
            'unique_words': len(word_freq)
        }
    
    def generate_summary_stats(self, df: pd.DataFrame) -> Dict:
        """Generate summary statistics"""
        stats = {
            'total_events': len(df),
            'sources': df['source'].value_counts().to_dict() if 'source' in df.columns else {},
            'regions': df['region'].value_counts().to_dict() if 'region' in df.columns else {},
            'risk_levels': df['risk_level'].value_counts().to_dict() if 'risk_level' in df.columns else {}
        }
        
        # Add conflict score stats
        if 'conflict_score' in df.columns:
            stats['conflict_score_stats'] = {
                'mean': float(df['conflict_score'].mean()),
                'median': float(df['conflict_score'].median()),
                'max': float(df['conflict_score'].max()),
                'min': float(df['conflict_score'].min())
            }
        
        # Add temporal stats if available
        time_cols = ['date', 'created_at', 'published_at']
        for col in time_cols:
            if col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col])
                    stats['time_range'] = {
                        'start': df[col].min().isoformat(),
                        'end': df[col].max().isoformat(),
                        'duration_days': (df[col].max() - df[col].min()).days
                    }
                    break
                except:
                    continue
        
        return stats
    
    def create_alert_report(self, high_risk_events: pd.DataFrame) -> Dict:
        """Create alert report for high-risk events"""
        if high_risk_events.empty:
            return {'alerts': [], 'summary': 'No high-risk alerts'}
        
        alerts = []
        
        for _, event in high_risk_events.iterrows():
            alert = {
                'id': str(event.get('id', hash(str(event)))),
                'timestamp': event.get('date', datetime.now()).isoformat(),
                'risk_level': event.get('risk_level', 'Unknown'),
                'location': event.get('region', 'Unknown'),
                'description': event.get('text', 'No description'),
                'confidence': event.get('confidence', 0.5),
                'source': event.get('source', 'Unknown'),
                'sentiment': event.get('vader_compound', 0),
                'conflict_score': event.get('conflict_score', 0)
            }
            alerts.append(alert)
        
        # Sort by risk level (Critical first)
        risk_order = {'Critical': 0, 'High': 1, 'Medium': 2, 'Low': 3}
        alerts.sort(key=lambda x: risk_order.get(x['risk_level'], 4))
        
        return {
            'alerts': alerts,
            'summary': {
                'total_alerts': len(alerts),
                'critical_count': len([a for a in alerts if a['risk_level'] == 'Critical']),
                'high_count': len([a for a in alerts if a['risk_level'] == 'High']),
                'generated_at': datetime.now().isoformat()
            }
        }

def test_visualizer():
    """Test the visualizer"""
    print("🎨 Testing Conflict Visualizer")
    print("=" * 50)
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=30, freq='D')
    
    sample_data = []
    for i in range(100):
        risk_level = np.random.choice(['Low', 'Medium', 'High', 'Critical'], 
                                     p=[0.5, 0.3, 0.15, 0.05])
        
        sample_data.append({
            'id': i,
            'date': np.random.choice(dates),
            'region': np.random.choice(['Nairobi', 'Mombasa', 'Kisumu', 'Nakuru']),
            'risk_level': risk_level,
            'conflict_score': np.random.uniform(0, 1),
            'vader_compound': np.random.uniform(-1, 1),
            'source': np.random.choice(['newsapi', 'reddit', 'gdelt']),
            'latitude': np.random.uniform(-4.0, 4.0),
            'longitude': np.random.uniform(33.0, 41.0),
            'text': f"Event in {risk_level} risk area",
            'cleaned_text': f"event {risk_level} risk area"
        })
    
    df = pd.DataFrame(sample_data)
    
    # Initialize visualizer
    visualizer = ConflictVisualizer()
    
    print("\n1. Creating dashboard...")
    dashboard = visualizer.create_risk_dashboard(df)
    
    print(f"   Generated {len(dashboard)} visualizations")
    
    print("\n2. Summary Statistics:")
    stats = dashboard['summary_stats']
    print(f"   Total Events: {stats['total_events']}")
    print(f"   Risk Distribution: {stats['risk_levels']}")
    
    print("\n3. Alert Report:")
    high_risk = df[df['risk_level'].isin(['High', 'Critical'])]
    alert_report = visualizer.create_alert_report(high_risk)
    print(f"   High Risk Alerts: {alert_report['summary']['total_alerts']}")
    
    print("\n" + "=" * 50)
    print("✅ Visualization testing complete!")

if __name__ == "__main__":
    test_visualizer()