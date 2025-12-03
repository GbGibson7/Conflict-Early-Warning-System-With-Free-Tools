"""
Conflict Early Warning System - Streamlit Dashboard
100% Free Deployment on Streamlit Cloud - FIXED VERSION
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
import os
import json

from functools import lru_cache

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_data():
    # Your data loading logic here
    pass

@st.cache_resource
def load_model():
    # Your model loading logic here
    pass

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules - UPDATED IMPORTS
try:
    # Use the simple collector instead of the Reddit-dependent one
    from src.data_collection.simple_collector import SimpleDataCollector
    from src.preprocessing.text_processor import TextProcessor
    from src.models.simple_predictor import SimpleConflictPredictor  # Changed to simple predictor
    from src.visualization.plot_generator import ConflictVisualizer
except ImportError as e:
    st.warning(f"Import error: {e}. Some features may be limited.")
    # Create mock imports for demo
    class MockModule:
        def __init__(self):
            pass
    
    SimpleDataCollector = MockModule
    TextProcessor = MockModule
    SimpleConflictPredictor = MockModule
    ConflictVisualizer = MockModule

# Page configuration
st.set_page_config(
    page_title="Conflict Early Warning System",
    page_icon="⚠️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #1E88E5;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1E88E5;
        margin-bottom: 1rem;
    }
    .risk-critical {
        background-color: #FF0000;
        color: white;
        padding: 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
    }
    .risk-high {
        background-color: #FF6B00;
        color: white;
        padding: 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
    }
    .risk-medium {
        background-color: #FFD700;
        color: black;
        padding: 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
    }
    .risk-low {
        background-color: #00FF00;
        color: black;
        padding: 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
    }
    .alert-box {
        background-color: #FFF3CD;
        border: 1px solid #FFEAA7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class ConflictDashboard:
    """Main dashboard class - FIXED VERSION"""
    
    def __init__(self):
        self.initialize_session_state()
        
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        if 'processed_data' not in st.session_state:
            st.session_state.processed_data = None
        if 'predictions' not in st.session_state:
            st.session_state.predictions = []
        if 'alerts' not in st.session_state:
            st.session_state.alerts = []
        if 'dashboard' not in st.session_state:
            st.session_state.dashboard = {}
        
    def render_sidebar(self):
        """Render sidebar controls"""
        with st.sidebar:
            st.image("https://img.icons8.com/color/96/000000/warning-shield.png", 
                    width=100)
            st.title("⚙️ Controls")
            
            st.subheader("Data Collection")
            query = st.text_input("Search Query", "conflict Kenya")
            days = st.slider("Days to collect", 1, 30, 7)
            
            col1, col2 = st.columns(2)
            with col1:
                collect_btn = st.button("📡 Collect Data", type="primary", use_container_width=True)
            with col2:
                load_demo = st.button("📊 Load Demo", use_container_width=True)
            
            st.divider()
            
            st.subheader("Analysis Settings")
            risk_threshold = st.slider("High Risk Threshold", 0.0, 1.0, 0.5)
            auto_refresh = st.checkbox("Auto-refresh (5 min)", False)
            
            st.divider()
            
            st.subheader("Export Data")
            if st.button("📥 Export CSV", use_container_width=True):
                self.export_data()
            
            if st.button("🔄 Clear Cache", use_container_width=True):
                st.session_state.clear()
                st.rerun()
            
            return query, days, collect_btn, load_demo, risk_threshold
    
    def collect_data(self, query: str, days: int):
        """Collect data from APIs - UPDATED for simple collector"""
        with st.spinner("📡 Collecting data from free APIs (No Reddit needed)..."):
            try:
                # Initialize SIMPLE collector (no Reddit dependency)
                collector = SimpleDataCollector()
                
                # Collect data
                data = collector.collect_all_no_reddit()
                
                if not data.empty:
                    # Process data with text processor
                    try:
                        processor = TextProcessor()
                        processed_data = processor.process_dataframe(data)
                    except Exception as e:
                        st.warning(f"Text processing skipped: {e}")
                        processed_data = data
                    
                    # Ensure required columns exist
                    processed_data = self._ensure_required_columns(processed_data)
                    
                    # Store in session
                    st.session_state.processed_data = processed_data
                    st.session_state.data_loaded = True
                    
                    st.success(f"✅ Collected {len(data)} records")
                    
                    # Show preview
                    with st.expander("📋 Data Preview"):
                        st.dataframe(processed_data.head(), use_container_width=True)
                    
                    # Auto-analyze
                    self.analyze_data()
                else:
                    st.error("❌ No data collected. Using demo data.")
                    self.load_demo_data()
                    
            except Exception as e:
                st.error(f"Error collecting data: {e}")
                st.info("Loading demo data instead...")
                self.load_demo_data()
    
    def _ensure_required_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure all required columns exist in the dataframe"""
        df = df.copy()
        
        # Ensure risk_level column exists
        if 'risk_level' not in df.columns:
            # Simple rule-based risk assignment
            def assign_risk(row):
                text = str(row.get('text', '') or row.get('title', '') or row.get('description', '')).lower()
                if any(kw in text for kw in ['terror', 'explosion', 'death', 'kill', 'bomb']):
                    return 'Critical'
                elif any(kw in text for kw in ['violence', 'attack', 'clash', 'riot']):
                    return 'High'
                elif any(kw in text for kw in ['protest', 'demonstration', 'strike', 'unrest']):
                    return 'Medium'
                else:
                    return 'Low'
            
            df['risk_level'] = df.apply(assign_risk, axis=1)
        
        # Ensure text column exists
        if 'text' not in df.columns:
            df['text'] = df.get('title', '') + ' ' + df.get('description', '')
        
        # Ensure numeric columns exist
        if 'conflict_score' not in df.columns:
            # Map risk level to conflict score
            risk_to_score = {'Low': 0.2, 'Medium': 0.5, 'High': 0.7, 'Critical': 0.9}
            df['conflict_score'] = df['risk_level'].map(risk_to_score).fillna(0.5)
        
        if 'vader_compound' not in df.columns:
            # Map risk level to sentiment
            risk_to_sentiment = {'Low': 0.3, 'Medium': -0.1, 'High': -0.5, 'Critical': -0.8}
            df['vader_compound'] = df['risk_level'].map(risk_to_sentiment).fillna(0)
        
        if 'has_conflict_keywords' not in df.columns:
            df['has_conflict_keywords'] = (df['risk_level'].isin(['High', 'Critical'])).astype(int)
        
        # Ensure source and region columns
        for col in ['source', 'region']:
            if col not in df.columns:
                df[col] = 'unknown'
        
        return df
    
    def load_demo_data(self):
        """Load demo data for testing - UPDATED with proper columns"""
        # Create synthetic demo data
        np.random.seed(42)
        n_samples = 200
        dates = pd.date_range('2024-01-01', periods=30, freq='D')
        
        demo_data = []
        for i in range(n_samples):
            risk_level = np.random.choice(['Low', 'Medium', 'High', 'Critical'],
                                          p=[0.6, 0.25, 0.1, 0.05])
            
            region = np.random.choice(['Nairobi', 'Mombasa', 'Kisumu', 'Nakuru', 'Eldoret'])
            
            # Generate realistic text based on risk level
            if risk_level == 'Critical':
                text = f"🚨 CRITICAL: Violent clashes reported in {region}. Emergency services responding."
                conflict_score = np.random.uniform(0.8, 1.0)
                sentiment = np.random.uniform(-1.0, -0.6)
            elif risk_level == 'High':
                text = f"⚠️ HIGH: Tensions rising in {region}. Protests turning violent."
                conflict_score = np.random.uniform(0.6, 0.8)
                sentiment = np.random.uniform(-0.8, -0.3)
            elif risk_level == 'Medium':
                text = f"🔶 MEDIUM: Demonstrations in {region}. Police monitoring situation."
                conflict_score = np.random.uniform(0.3, 0.6)
                sentiment = np.random.uniform(-0.5, 0.1)
            else:
                text = f"✅ LOW: Peaceful activities in {region}. Normal operations."
                conflict_score = np.random.uniform(0.0, 0.3)
                sentiment = np.random.uniform(-0.2, 0.5)
            
            demo_data.append({
                'id': i + 1,
                'date': np.random.choice(dates),
                'region': region,
                'text': text,
                'cleaned_text': text.lower(),
                'risk_level': risk_level,
                'conflict_score': conflict_score,
                'vader_compound': sentiment,
                'vader_negative': max(-sentiment, 0) if sentiment < 0 else 0,
                'vader_positive': max(sentiment, 0) if sentiment > 0 else 0,
                'source': np.random.choice(['newsapi', 'gdelt', 'synthetic']),  # No reddit
                'word_count': len(text.split()),
                'has_conflict_keywords': 1 if conflict_score > 0.3 else 0,
                'latitude': np.random.uniform(-4.0, 4.0),
                'longitude': np.random.uniform(33.0, 41.0)
            })
        
        df = pd.DataFrame(demo_data)
        
        # Store in session
        st.session_state.processed_data = df
        st.session_state.data_loaded = True
        
        st.success(f"✅ Loaded {len(df)} demo records")
        
        # Show preview
        with st.expander("📋 Demo Data Preview"):
            st.dataframe(df[['date', 'region', 'risk_level', 'text']].head(10), 
                        use_container_width=True)
        
        # Auto-analyze
        self.analyze_data()
    
    def analyze_data(self):
        """Analyze the collected data - FIXED VERSION"""
        if not st.session_state.data_loaded or st.session_state.processed_data is None:
            st.warning("⚠️ No data loaded. Please collect data first.")
            return
        
        data = st.session_state.processed_data.copy()  # Make a copy
        
        with st.spinner("🔍 Analyzing data..."):
            try:
                # Ensure required columns exist
                data = self._ensure_required_columns(data)
                
                # Use the SIMPLE predictor instead of the complex one
                predictor = SimpleConflictPredictor()
                
                # Store updated data
                st.session_state.processed_data = data
                
                # Initialize visualizer
                visualizer = ConflictVisualizer()
                
                # Create dashboard visualizations
                dashboard = visualizer.create_risk_dashboard(data)
                
                # Store visualizations
                st.session_state.dashboard = dashboard
                
                # Generate alerts
                if 'risk_level' in data.columns:
                    high_risk = data[data['risk_level'].isin(['High', 'Critical'])]
                    st.session_state.alerts = visualizer.create_alert_report(high_risk)
                else:
                    st.session_state.alerts = {'alerts': [], 'summary': 'No risk data'}
                
                st.success("✅ Analysis complete!")
                
            except Exception as e:
                st.error(f"Analysis error: {e}")
                st.info("Using basic visualization only...")
                
                # Basic visualization fallback
                try:
                    visualizer = ConflictVisualizer()
                    st.session_state.dashboard = visualizer.create_risk_dashboard(data)
                except Exception as viz_error:
                    st.error(f"Visualization error: {viz_error}")
                    st.session_state.dashboard = {}
    
    def render_dashboard(self):
        """Render the main dashboard - FIXED with error handling"""
        st.markdown('<div class="main-header">⚠️ Conflict Early Warning System</div>', 
                   unsafe_allow_html=True)
        
        # Top metrics row - WITH ERROR HANDLING
        col1, col2, col3, col4 = st.columns(4)
        
        if st.session_state.data_loaded and st.session_state.processed_data is not None:
            data = st.session_state.processed_data
            
            # Safely calculate metrics with error handling
            try:
                total_events = len(data)
            except:
                total_events = 0
            
            try:
                if 'risk_level' in data.columns:
                    high_risk = len(data[data['risk_level'].isin(['High', 'Critical'])])
                else:
                    high_risk = 0
            except:
                high_risk = 0
            
            try:
                if 'conflict_score' in data.columns:
                    avg_conflict = data['conflict_score'].mean()
                    avg_conflict_str = f"{avg_conflict:.2f}"
                else:
                    avg_conflict_str = "N/A"
            except:
                avg_conflict_str = "N/A"
            
            try:
                date_col = None
                for col in ['date', 'created_at', 'published_at']:
                    if col in data.columns:
                        date_col = col
                        break
                
                if date_col:
                    latest_date = pd.to_datetime(data[date_col]).max()
                    latest_date_str = latest_date.strftime('%Y-%m-%d')
                else:
                    latest_date_str = "N/A"
            except:
                latest_date_str = "N/A"
            
            with col1:
                st.metric("📊 Total Events", total_events)
            with col2:
                st.metric("🚨 High Risk Events", high_risk)
            with col3:
                st.metric("📈 Avg Conflict Score", avg_conflict_str)
            with col4:
                st.metric("⏰ Latest Data", latest_date_str)
        else:
            # Show placeholders if no data
            with col1:
                st.metric("📊 Total Events", 0)
            with col2:
                st.metric("🚨 High Risk Events", 0)
            with col3:
                st.metric("📈 Avg Conflict Score", "N/A")
            with col4:
                st.metric("⏰ Latest Data", "N/A")
        
        # Main dashboard tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🗺️ Map", "📈 Trends", "🚨 Alerts"])
        
        with tab1:
            self.render_overview_tab()
        
        with tab2:
            self.render_map_tab()
        
        with tab3:
            self.render_trends_tab()
        
        with tab4:
            self.render_alerts_tab()
    
    def render_overview_tab(self):
        """Render overview tab - FIXED with error handling"""
        if not st.session_state.data_loaded:
            st.info("👈 Collect data from the sidebar to begin")
            return
        
        if 'dashboard' not in st.session_state or not st.session_state.dashboard:
            st.warning("Please run analysis first or data has issues")
            if st.button("🔄 Run Analysis Now"):
                self.analyze_data()
                st.rerun()
            return
        
        dashboard = st.session_state.dashboard
        
        # Row 1: Risk Distribution and Summary
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown('<div class="sub-header">Risk Level Distribution</div>', 
                       unsafe_allow_html=True)
            
            if 'risk_distribution' in dashboard and dashboard['risk_distribution']:
                risk_data = dashboard['risk_distribution'].get('data', {})
                
                if risk_data:
                    # Create a Plotly pie chart
                    fig = go.Figure(data=[go.Pie(
                        labels=list(risk_data.keys()),
                        values=list(risk_data.values()),
                        hole=.3,
                        marker_colors=['#00FF00', '#FFD700', '#FF6B00', '#FF0000']
                    )])
                    
                    fig.update_layout(
                        height=400,
                        showlegend=True,
                        margin=dict(t=0, b=0, l=0, r=0)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No risk distribution data available")
            else:
                st.info("Risk distribution not calculated yet")
        
        with col2:
            st.markdown('<div class="sub-header">Quick Stats</div>', 
                       unsafe_allow_html=True)
            
            if 'summary_stats' in dashboard:
                stats = dashboard['summary_stats']
                
                # Display metrics in cards
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Total Events", stats.get('total_events', 0))
                st.markdown('</div>', unsafe_allow_html=True)
                
                if 'risk_levels' in stats:
                    for level, count in stats['risk_levels'].items():
                        color_class = f"risk-{level.lower()}"
                        st.markdown(f'<div class="{color_class}">{level}: {count}</div>', 
                                  unsafe_allow_html=True)
                else:
                    st.info("No risk level statistics")
            else:
                st.info("No summary statistics available")
        
        # Row 2: Sentiment Analysis
        st.markdown('<div class="sub-header">Sentiment Analysis</div>', 
                   unsafe_allow_html=True)
        
        if 'sentiment_analysis' in dashboard and dashboard['sentiment_analysis']:
            sentiment_stats = dashboard['sentiment_analysis'].get('statistics', {})
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("😊 Positive", f"{sentiment_stats.get('positive_percentage', 0):.1f}%")
            with col2:
                st.metric("😐 Neutral", f"{sentiment_stats.get('neutral_percentage', 0):.1f}%")
            with col3:
                st.metric("😠 Negative", f"{sentiment_stats.get('negative_percentage', 0):.1f}%")
            with col4:
                st.metric("📊 Avg Sentiment", f"{sentiment_stats.get('mean', 0):.2f}")
        else:
            st.info("Sentiment analysis not available")
        
        # Row 3: Top Keywords
        st.markdown('<div class="sub-header">Top Keywords</div>', 
                   unsafe_allow_html=True)
        
        if 'word_cloud_data' in dashboard and dashboard['word_cloud_data']:
            word_freq = dashboard['word_cloud_data'].get('word_frequencies', {})
            
            if word_freq:
                # Create bar chart of top keywords
                top_words = list(word_freq.keys())[:15]
                frequencies = [word_freq[word] for word in top_words]
                
                fig = go.Figure(data=[go.Bar(
                    x=frequencies,
                    y=top_words,
                    orientation='h',
                    marker_color='#1E88E5'
                )])
                
                fig.update_layout(
                    height=400,
                    xaxis_title="Frequency",
                    yaxis_title="Keywords",
                    margin=dict(t=20, b=20, l=100, r=20)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No keyword data available")
        else:
            st.info("Word cloud data not calculated yet")
    
    def render_map_tab(self):
        """Render map visualization tab - FIXED"""
        if not st.session_state.data_loaded:
            st.info("👈 Collect data from the sidebar to begin")
            return
        
        st.markdown('<div class="sub-header">Geographic Heatmap</div>', 
                   unsafe_allow_html=True)
        
        if st.session_state.processed_data is not None:
            data = st.session_state.processed_data
            
            # Check if we have coordinates
            if 'latitude' in data.columns and 'longitude' in data.columns:
                # Filter valid coordinates
                map_data = data.dropna(subset=['latitude', 'longitude'])
                map_data = map_data[
                    (map_data['latitude'].between(-90, 90)) & 
                    (map_data['longitude'].between(-180, 180))
                ]
                
                if not map_data.empty:
                    # Simple Streamlit map
                    st.map(map_data[['latitude', 'longitude']], use_container_width=True)
                    
                    # Show location stats
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Events by Region:**")
                        if 'region' in map_data.columns:
                            region_counts = map_data['region'].value_counts()
                            for region, count in region_counts.head(10).items():
                                st.write(f"{region}: {count}")
                    
                    with col2:
                        st.write("**Risk by Region:**")
                        if 'region' in map_data.columns and 'risk_level' in map_data.columns:
                            region_risk = map_data.groupby('region')['risk_level'].apply(
                                lambda x: (x.isin(['High', 'Critical'])).sum()
                            ).sort_values(ascending=False)
                            
                            for region, count in region_risk.head(10).items():
                                if count > 0:
                                    st.write(f"{region}: {count} high-risk events")
                else:
                    st.info("No valid geographic coordinates available")
            else:
                st.info("No geographic data available. Collect data with location information.")
        else:
            st.info("No data available for mapping")
    
    def render_trends_tab(self):
        """Render trends analysis tab - SIMPLIFIED WORKING VERSION"""
        if not st.session_state.data_loaded:
            st.info("👈 Collect data from the sidebar to begin")
            return
        
        if st.session_state.processed_data is None:
            st.info("No data available")
            return
        
        data = st.session_state.processed_data.copy()
        
        st.markdown('<div class="sub-header">📈 Temporal Trends</div>', 
                   unsafe_allow_html=True)
        
        # Find date column
        date_col = None
        for col in ['date', 'created_at', 'published_at', 'collected_at', 'seendate']:
            if col in data.columns:
                date_col = col
                break
        
        if not date_col:
            st.info("No date column found for trend analysis")
            return
        
        try:
            # Try to parse dates
            data['parsed_date'] = pd.to_datetime(data[date_col], errors='coerce')
            data_clean = data.dropna(subset=['parsed_date'])
            
            if data_clean.empty:
                st.info("No valid dates found for trend analysis")
                return
            
            # Extract date part
            data_clean['date_only'] = data_clean['parsed_date'].dt.date
            
            # Sort by date
            data_clean = data_clean.sort_values('date_only')
            
            # Create two columns for charts
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Conflict Score Trends")
                if 'conflict_score' in data_clean.columns:
                    # Group by date
                    daily_conflict = data_clean.groupby('date_only')['conflict_score'].agg(['mean', 'count']).reset_index()
                    
                    if not daily_conflict.empty:
                        # SIMPLE line chart (no dual axis)
                        fig = px.line(
                            daily_conflict,
                            x='date_only',
                            y='mean',
                            title='Daily Average Conflict Score',
                            labels={'date_only': 'Date', 'mean': 'Avg Conflict Score'}
                        )
                        
                        # Customize
                        fig.update_traces(
                            line=dict(color='#FF4B4B', width=3),
                            mode='lines+markers',
                            marker=dict(size=8, color='#FF4B4B')
                        )
                        
                        fig.update_layout(
                            height=400,
                            hovermode='x unified',
                            xaxis_title='Date',
                            yaxis_title='Conflict Score (0-1)',
                            yaxis_range=[0, 1]
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show event count separately
                        st.caption(f"📅 Total events analyzed: {len(data_clean)}")
                        avg_score = daily_conflict['mean'].mean()
                        st.metric("Overall Avg Score", f"{avg_score:.2f}")
                    else:
                        st.info("No conflict score data available")
                else:
                    st.info("Conflict score column not found")
            
            with col2:
                st.subheader("⚠️ Risk Level Distribution")
                if 'risk_level' in data_clean.columns:
                    # Create pivot table for risk levels
                    risk_pivot = pd.crosstab(data_clean['date_only'], data_clean['risk_level'])
                    
                    if not risk_pivot.empty:
                        # Ensure all risk levels are present
                        for risk in ['Low', 'Medium', 'High', 'Critical']:
                            if risk not in risk_pivot.columns:
                                risk_pivot[risk] = 0
                        
                        # Reorder columns
                        risk_pivot = risk_pivot[['Low', 'Medium', 'High', 'Critical']]
                        
                        # Create simple line chart for each risk level
                        fig = go.Figure()
                        
                        colors = {
                            'Low': '#00C851',
                            'Medium': '#FFBB33',
                            'High': '#FF8800',
                            'Critical': '#CC0000'
                        }
                        
                        # Add trace for each risk level
                        for risk_level in ['Low', 'Medium', 'High', 'Critical']:
                            if risk_level in risk_pivot.columns:
                                fig.add_trace(go.Scatter(
                                    x=risk_pivot.index,
                                    y=risk_pivot[risk_level],
                                    mode='lines+markers',
                                    name=risk_level,
                                    line=dict(width=2, color=colors.get(risk_level)),
                                    marker=dict(size=6)
                                ))
                        
                        fig.update_layout(
                            title='Risk Level Trends Over Time',
                            xaxis_title='Date',
                            yaxis_title='Number of Events',
                            hovermode='x unified',
                            height=400,
                            showlegend=True
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Risk level statistics
                        st.markdown("**Risk Level Summary:**")
                        risk_totals = data_clean['risk_level'].value_counts()
                        
                        # Display in a nice grid
                        for risk_level in ['Critical', 'High', 'Medium', 'Low']:
                            if risk_level in risk_totals:
                                count = risk_totals[risk_level]
                                percentage = (count / len(data_clean)) * 100
                                color_class = f"risk-{risk_level.lower()}"
                                st.markdown(
                                    f'<div class="{color_class}" style="padding: 0.5rem; margin: 0.25rem 0; border-radius: 0.25rem;">'
                                    f'{risk_level}: {count} events ({percentage:.1f}%)'
                                    f'</div>',
                                    unsafe_allow_html=True
                                )
                    else:
                        st.info("No risk level data available")
                else:
                    st.info("Risk level column not found")
            
            # Date range and statistics
            st.markdown("---")
            
            # Date information in columns
            date_col1, date_col2, date_col3 = st.columns(3)
            
            with date_col1:
                earliest = data_clean['date_only'].min()
                st.metric("📅 Earliest Date", str(earliest))
            
            with date_col2:
                latest = data_clean['date_only'].max()
                st.metric("📅 Latest Date", str(latest))
            
            with date_col3:
                date_range = (latest - earliest).days
                st.metric("📊 Date Range", f"{date_range} days")
            
            # Additional insights
            st.markdown("---")
            st.subheader("📈 Quick Insights")
            
            insight_col1, insight_col2, insight_col3 = st.columns(3)
            
            with insight_col1:
                if 'conflict_score' in data_clean.columns:
                    max_conflict = data_clean['conflict_score'].max()
                    st.metric("Peak Conflict", f"{max_conflict:.2f}")
            
            with insight_col2:
                if 'risk_level' in data_clean.columns:
                    high_risk_count = (data_clean['risk_level'].isin(['High', 'Critical'])).sum()
                    st.metric("High/Critical Events", high_risk_count)
            
            with insight_col3:
                total_events = len(data_clean)
                st.metric("Total Events", total_events)
            
            # Data preview
            with st.expander("🔍 View Sample Data (10 records)"):
                preview_cols = ['date_only', 'risk_level', 'conflict_score', 'source', 'region']
                available_cols = [col for col in preview_cols if col in data_clean.columns]
                
                if available_cols:
                    st.dataframe(
                        data_clean[available_cols].head(10),
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.info("No preview data available")
            
        except Exception as e:
            st.error(f"Error creating trends: {str(e)}")
            
            # Show helpful debug info
            with st.expander("🛠️ Debug Information"):
                st.write(f"**Error:** {type(e).__name__}")
                st.write(f"**Message:** {str(e)}")
                
                if date_col in data.columns:
                    st.write(f"**Date column:** {date_col}")
                    st.write("**Sample values:**")
                    sample = data[date_col].dropna().head(3).tolist()
                    for val in sample:
                        st.code(f"{val}")
                
                st.write("**Available columns:**")
                st.write(list(data.columns))
            
            st.info("Please check your data format or try loading demo data.")
    
    def render_alerts_tab(self):
        """Render alerts tab - FIXED with simple predictor"""
        if not st.session_state.data_loaded:
            st.info("👈 Collect data from the sidebar to begin")
            return
        
        st.markdown('<div class="sub-header">🚨 Active Alerts</div>', 
                   unsafe_allow_html=True)
        
        if 'alerts' in st.session_state and st.session_state.alerts:
            alerts_data = st.session_state.alerts
            
            if alerts_data.get('summary', {}).get('total_alerts', 0) > 0:
                st.markdown(f"""
                <div class="alert-box">
                <h4>📢 Alert Summary</h4>
                <p>Total Alerts: <strong>{alerts_data['summary']['total_alerts']}</strong></p>
                <p>Critical: <span class="risk-critical">{alerts_data['summary'].get('critical_count', 0)}</span></p>
                <p>High: <span class="risk-high">{alerts_data['summary'].get('high_count', 0)}</span></p>
                <p>Generated: {alerts_data['summary'].get('generated_at', 'Unknown')}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Display individual alerts
                for alert in alerts_data.get('alerts', []):
                    risk_class = f"risk-{alert.get('risk_level', 'Medium').lower()}"
                    
                    st.markdown(f"""
                    <div style="border-left: 4px solid {self.get_risk_color(alert.get('risk_level', 'Medium'))}; 
                                padding: 1rem; margin: 1rem 0; background-color: #f8f9fa;">
                        <div style="display: flex; justify-content: space-between;">
                            <h5 style="margin: 0;">{alert.get('location', 'Unknown')} - {alert.get('risk_level', 'Unknown')}</h5>
                            <span class="{risk_class}">{alert.get('risk_level', 'Unknown')}</span>
                        </div>
                        <p style="margin: 0.5rem 0; color: #666;">{alert.get('timestamp', 'Unknown')}</p>
                        <p style="margin: 0.5rem 0;">{alert.get('description', 'No description')}</p>
                        <div style="display: flex; gap: 1rem; font-size: 0.9rem;">
                            <span>Confidence: {alert.get('confidence', 0):.1%}</span>
                            <span>Source: {alert.get('source', 'Unknown')}</span>
                            <span>Sentiment: {alert.get('sentiment', 0):.2f}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.success("✅ No active alerts. Situation is stable.")
        else:
            st.info("No alerts generated yet. Run analysis to generate alerts.")
        
        # Alert prediction form
        st.markdown('<div class="sub-header">🔮 Predict Risk for New Text</div>', 
                   unsafe_allow_html=True)
        
        with st.form("prediction_form"):
            new_text = st.text_area("Enter text to analyze:", 
                                   "Violent protests reported in downtown area")
            col1, col2 = st.columns(2)
            with col1:
                region = st.selectbox("Region:", 
                                     ['Nairobi', 'Mombasa', 'Kisumu', 'Nakuru', 'Eldoret', 'Other'])
            with col2:
                source = st.selectbox("Source:", 
                                     ['news', 'social media', 'official report', 'eyewitness'])
            
            predict_btn = st.form_submit_button("🔍 Predict Risk", type="primary")
            
            if predict_btn and new_text:
                with st.spinner("Analyzing..."):
                    try:
                        # Initialize SIMPLE predictor
                        predictor = SimpleConflictPredictor()
                        
                        # Make prediction
                        prediction = predictor.predict_risk(new_text, {'region': region, 'source': source})
                        
                        # Display result
                        risk_level = prediction.get('risk_level', 'Medium')
                        action = prediction.get('recommended_action', 'Monitor situation')
                        confidence = prediction.get('confidence', 0.7)
                        risk_class = f"risk-{risk_level.lower()}"
                        
                        st.markdown(f"""
                        <div style="padding: 1.5rem; border-radius: 0.5rem; 
                                    background-color: {'#FFEBEE' if risk_level in ['High', 'Critical'] else '#E8F5E9'}; 
                                    margin: 1rem 0;">
                            <h3>Prediction Result:</h3>
                            <p><strong>Risk Level:</strong> <span class="{risk_class}">{risk_level}</span> ({confidence:.1%} confidence)</p>
                            <p><strong>Recommended Action:</strong> {action}</p>
                            <p><strong>Analysis:</strong> {new_text[:100]}...</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"Prediction error: {e}")
                        # Fallback rule-based prediction
                        text_lower = new_text.lower()
                        if any(kw in text_lower for kw in ['terror', 'explosion', 'death']):
                            risk_level = 'Critical'
                        elif any(kw in text_lower for kw in ['violence', 'attack', 'clash']):
                            risk_level = 'High'
                        elif any(kw in text_lower for kw in ['protest', 'demonstration']):
                            risk_level = 'Medium'
                        else:
                            risk_level = 'Low'
                        
                        st.success(f"Rule-based prediction: **{risk_level}** risk level")
    
    def get_risk_color(self, risk_level: str) -> str:
        """Get color for risk level"""
        color_map = {
            'Critical': '#FF0000',
            'High': '#FF6B00',
            'Medium': '#FFD700',
            'Low': '#00FF00'
        }
        return color_map.get(risk_level, '#CCCCCC')
    
    def export_data(self):
        """Export data as CSV"""
        if st.session_state.data_loaded and st.session_state.processed_data is not None:
            data = st.session_state.processed_data
            csv = data.to_csv(index=False)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"conflict_data_{timestamp}.csv"
            
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=filename,
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.warning("No data to export. Please collect data first.")
    
    def run(self):
        """Run the dashboard"""
        # Sidebar
        query, days, collect_btn, load_demo, risk_threshold = self.render_sidebar()
        
        # Handle data collection
        if collect_btn:
            self.collect_data(query, days)
        
        if load_demo:
            self.load_demo_data()
        
        # Main dashboard
        self.render_dashboard()

def main():
    """Main function"""
    st.sidebar.title("🌍 Conflict Early Warning")
    st.sidebar.markdown("""
    ### About
    This system detects and predicts conflict risks using free data sources and machine learning.
    
    ### Features:
    - 📡 Real-time data collection (No Reddit needed!)
    - 🤖 AI-powered risk prediction
    - 🗺️ Geographic visualization
    - 🚨 Early warning alerts
    
    ### Free Tools Used:
    - NewsAPI, GDELT, Wikipedia
    - Streamlit Cloud (Deployment)
    - scikit-learn (ML)
    - Plotly (Visualization)
    
    ### Quick Start:
    1. Click "📊 Load Demo" for instant demo
    2. Or get NewsAPI key for real data
    3. Click "📡 Collect Data" for analysis
    """)
    
    # Initialize and run dashboard
    dashboard = ConflictDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()