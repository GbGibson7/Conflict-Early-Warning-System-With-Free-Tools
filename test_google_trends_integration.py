#!/usr/bin/env python3
"""
Test Google Trends integration with Conflict Early Warning System
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_collection.google_trends_collector import GoogleTrendsCollector
from data_collection.simple_collector import SimpleDataCollector
from models.simple_predictor import SimpleConflictPredictor

def test_integration():
    print("🧪 Testing Google Trends Integration")
    print("=" * 60)
    
    # Test 1: Google Trends Collector
    print("\n1. Testing Google Trends Collector...")
    trends_collector = GoogleTrendsCollector()
    trends_data = trends_collector.collect_all_trends_data()
    
    print(f"   ✓ Interest data: {len(trends_data['interest_over_time'])} records")
    print(f"   ✓ Trending searches: {len(trends_data['trending_searches'])}")
    print(f"   ✓ Risk score: {trends_data['risk_score']:.2f}")
    
    # Test 2: Integration with Simple Collector
    print("\n2. Testing integration with Simple Collector...")
    simple_collector = SimpleDataCollector()
    
    # Mock Google Trends data for testing
    mock_trends_data = trends_collector.format_for_main_system(trends_data)
    
    if not mock_trends_data.empty:
        print(f"   ✓ Formatted {len(mock_trends_data)} records for main system")
        print(f"   ✓ Sample: {mock_trends_data.iloc[0]['title'][:50]}...")
    else:
        print("   ⚠️ Could not format data (may be rate limited)")
    
    # Test 3: ML Prediction with Google Trends
    print("\n3. Testing ML prediction with Google Trends...")
    predictor = SimpleConflictPredictor()
    
    test_cases = [
        ("Violent protests in Nairobi", {'region': 'Nairobi'}),
        ("Peaceful demonstrations in Mombasa", {'region': 'Mombasa'}),
        ("Security alert in Kisumu", {'region': 'Kisumu'})
    ]
    
    for text, features in test_cases:
        prediction = predictor.predict_risk(text, features)
        print(f"   📝 '{text[:30]}...'")
        print(f"     Risk: {prediction['risk_level']}, Confidence: {prediction.get('confidence', 0):.2f}")
        if prediction.get('has_google_trends'):
            print(f"     Google Trends impact: {prediction.get('google_trends_impact', 0):.2f}")
    
    print("\n" + "=" * 60)
    print("✅ Integration test complete!")
    print("\n📊 **Next Steps:**")
    print("1. Run the dashboard: streamlit run dashboard/app.py")
    print("2. Click the '🌐 Google Trends' tab")
    print("3. Collect data and analyze trends")

if __name__ == "__main__":
    test_integration()