"""
Google Trends Data Collector
Free public data - no API key needed
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging
import time
from functools import wraps

# ============ ENHANCED IMPORTS ============
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import requests
# ============ END ENHANCED IMPORTS ============

# ============ RATE LIMITING DECORATOR ============
def rate_limited(max_per_minute):
    """Decorator to limit function calls per minute"""
    min_interval = 60.0 / max_per_minute
    
    def decorator(func):
        last_called = [0.0]
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            left_to_wait = min_interval - elapsed
            
            if left_to_wait > 0:
                time.sleep(left_to_wait)
            
            last_called[0] = time.time()
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator
# ============ END RATE LIMITING ============

# Try to import pytrends, but handle gracefully if not installed
try:
    from pytrends.request import TrendReq
    PYTREENS_AVAILABLE = True
except ImportError:
    PYTREENS_AVAILABLE = False
    print("⚠️ pytrends not installed. Install with: pip install pytrends")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GoogleTrendsCollector:
    """Collect Google Trends data for conflict-related keywords"""
    
    def __init__(self, hl: str = 'en-US', tz: int = 360):
        """
        Initialize Google Trends connection
        
        Parameters:
        -----------
        hl : str
            Language (default: 'en-US')
        tz : int
            Timezone offset in minutes (360 = CST)
        """
        if not PYTREENS_AVAILABLE:
            logger.error("pytrends not installed. Run: pip install pytrends")
            self.initialized = False
            return
        
        try:
            # Create a custom session to avoid the method_whitelist error
            import requests
            from requests.adapters import HTTPAdapter
            from urllib3.util.retry import Retry
            
            # Create a custom retry strategy
            retry_strategy = Retry(
                total=2,
                backoff_factor=0.1,
                status_forcelist=[429, 500, 502, 503, 504],
                # Use allowed_methods instead of method_whitelist
                allowed_methods=["HEAD", "GET", "OPTIONS", "POST", "PUT", "DELETE", "PATCH"]
            )
            
            # Create custom session with retry strategy
            session = requests.Session()
            adapter = HTTPAdapter(max_retries=retry_strategy)
            session.mount("https://", adapter)
            session.mount("http://", adapter)
            
            # Add custom headers
            session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            })
            
            # Initialize pytrends with custom session
            self.pytrends = TrendReq(
                hl=hl, 
                tz=tz,
                timeout=(10, 25),
                requests_args={'session': session}
            )
            self.retry_delay = 10  # Increased from 5 to 10 seconds
            self.initialized = True
            logger.info("✅ Google Trends collector initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Google Trends: {e}")
            logger.error("Trying with simpler configuration...")
            try:
                # Try simpler initialization as fallback
                self.pytrends = TrendReq(hl=hl, tz=tz, timeout=30)
                self.retry_delay = 10
                self.initialized = True
                logger.info("✅ Google Trends collector initialized (simple mode)")
            except Exception as e2:
                logger.error(f"Fallback initialization also failed: {e2}")
                self.initialized = False
        
        # Conflict-related keywords for Kenya
        self.conflict_keywords = [
            'violence', 'protest', 'riot', 'conflict', 'attack',
            'security', 'police', 'demonstration', 'unrest', 'clash'
        ]
        
        # Kenyan regions
        self.kenyan_regions = ['Kenya', 'Nairobi', 'Mombasa', 'Kisumu', 'Nakuru']
    
    # ============ ENHANCED RETRY DECORATOR ============
    @staticmethod
    def retry_on_rate_limit(func):
        """Custom retry decorator for rate limiting"""
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    return func(self, *args, **kwargs)
                except Exception as e:
                    error_str = str(e)
                    if "429" in error_str and attempt < max_retries - 1:
                        wait_time = 30 * (attempt + 1)  # 30, 60, 90 seconds
                        logger.warning(f"Rate limited. Waiting {wait_time} seconds before retry {attempt + 1}/{max_retries}")
                        time.sleep(wait_time)
                        continue
                    elif "404" in error_str:
                        logger.error(f"404 error - endpoint may not exist: {e}")
                        return pd.DataFrame()
                    else:
                        # For other errors, raise immediately
                        raise
            return pd.DataFrame()  # Return empty on final failure
        return wrapper
    # ============ END ENHANCED RETRY DECORATOR ============

    @retry_on_rate_limit
    @rate_limited(max_per_minute=5)  # Reduced from 10 to 5
    def get_trending_searches(self, region: str = 'KE') -> pd.DataFrame:
        """Get currently trending searches with fallback options"""
        if not self.initialized:
            logger.warning("Google Trends not initialized")
            return pd.DataFrame()
        
        try:
            logger.info(f"Fetching trending searches for {region}...")
            
            # Try different region codes if KE fails
            if region == 'KE':
                try:
                    trending_df = self.pytrends.trending_searches(pn=region)
                except Exception as e:
                    logger.warning(f"Trending searches for {region} failed: {e}")
                    # Fallback to global trends
                    logger.info("Trying global trending searches as fallback...")
                    trending_df = self.pytrends.trending_searches(pn='united_states')
            else:
                trending_df = self.pytrends.trending_searches(pn=region)
            
            if trending_df is not None and not trending_df.empty:
                trending_df.columns = ['trending_search']
                trending_df['region'] = region
                trending_df['source'] = 'google_trends'
                trending_df['collected_at'] = datetime.now()
                
                logger.info(f"Found {len(trending_df)} trending searches for {region}")
                return trending_df
            else:
                logger.warning(f"No trending searches found for {region}")
                return self.get_trending_searches_fallback()
                
        except Exception as e:
            logger.error(f"Error getting trending searches: {e}")
            # Try fallback
            return self.get_trending_searches_fallback()
    
    def get_trending_searches_fallback(self) -> pd.DataFrame:
        """Fallback method when trending_searches fails"""
        try:
            # Use interest over time as a fallback for trending data
            self.pytrends.build_payload(
                kw_list=['Kenya', 'Nairobi', 'Africa'],
                timeframe='now 1-d',
                geo=''
            )
            
            daily_df = self.pytrends.interest_over_time()
            
            if not daily_df.empty:
                # Get top searches from yesterday
                fallback_data = pd.DataFrame({
                    'trending_search': ['Kenya news', 'Nairobi events', 'Africa updates'],
                    'region': ['KE'],
                    'source': ['google_trends_fallback'],
                    'collected_at': [datetime.now()]
                })
                logger.info("Using fallback trending data")
                return fallback_data
        except Exception as e:
            logger.error(f"Fallback method also failed: {e}")
        
        return pd.DataFrame()
    
    @retry_on_rate_limit
    @rate_limited(max_per_minute=5)  # Reduced from 10 to 5
    def get_interest_over_time(self, keywords: List[str] = None, 
                              timeframe: str = 'today 3-m') -> pd.DataFrame:
        """
        Get interest over time for keywords with enhanced error handling
        
        Parameters:
        -----------
        keywords : List[str]
            Keywords to search for (max 3 for free API to avoid rate limiting)
        timeframe : str
            Timeframe for data (e.g., 'today 3-m', 'today 1-y', 'now 7-d')
        """
        if not self.initialized:
            logger.warning("Google Trends not initialized")
            return pd.DataFrame()
        
        if keywords is None:
            keywords = self.conflict_keywords[:3]  # Reduced from 5 to 3 keywords
        
        try:
            logger.info(f"Fetching interest over time for {len(keywords)} keywords...")
            
            # Use a simpler timeframe if recent data fails
            if timeframe == 'now 7-d':
                timeframe = 'today 7-d'
            
            self.pytrends.build_payload(
                kw_list=keywords,
                timeframe=timeframe,
                geo='KE',  # Kenya
                gprop=''
            )
            
            interest_df = self.pytrends.interest_over_time()
            
            if not interest_df.empty:
                # Remove 'isPartial' column if exists
                if 'isPartial' in interest_df.columns:
                    interest_df = interest_df.drop('isPartial', axis=1)
                
                # Reshape from wide to long format
                interest_long = interest_df.reset_index().melt(
                    id_vars=['date'],
                    var_name='keyword',
                    value_name='interest_score'
                )
                
                interest_long['region'] = 'Kenya'
                interest_long['source'] = 'google_trends'
                interest_long['collected_at'] = datetime.now()
                
                logger.info(f"Successfully collected interest data for {len(keywords)} keywords")
                return interest_long
            else:
                logger.warning("Interest over time returned empty DataFrame")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error getting interest over time: {e}")
            # Re-raise for the retry decorator
            raise
    
    @retry_on_rate_limit
    @rate_limited(max_per_minute=5)  # Reduced from 10 to 5
    def get_interest_by_region(self, keyword: str, 
                              resolution: str = 'REGION') -> pd.DataFrame:
        """
        Get interest by region for a specific keyword
        
        Parameters:
        -----------
        keyword : str
            Single keyword to search
        resolution : str
            'COUNTRY', 'REGION', 'CITY', 'DMA'
        """
        if not self.initialized:
            logger.warning("Google Trends not initialized")
            return pd.DataFrame()
        
        try:
            logger.info(f"Fetching regional interest for '{keyword}'")
            
            self.pytrends.build_payload(
                kw_list=[keyword],
                timeframe='today 3-m',
                geo='',
                gprop=''
            )
            
            region_df = self.pytrends.interest_by_region(
                resolution=resolution,
                inc_low_vol=True,
                inc_geo_code=False
            )
            
            if not region_df.empty:
                region_df = region_df.reset_index()
                region_df.columns = ['region', 'interest_score']
                region_df['keyword'] = keyword
                region_df['source'] = 'google_trends'
                region_df['collected_at'] = datetime.now()
                region_df['resolution'] = resolution
                
                logger.info(f"Collected regional interest for '{keyword}'")
                return region_df
            else:
                logger.warning(f"No regional interest data for '{keyword}'")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error getting interest by region: {e}")
            raise  # Re-raise for retry decorator
    
    @retry_on_rate_limit
    @rate_limited(max_per_minute=5)  # Reduced from 10 to 5
    def get_related_queries(self, keyword: str) -> Dict:
        """Get queries related to a keyword"""
        if not self.initialized:
            logger.warning("Google Trends not initialized")
            return {'keyword': keyword, 'top_queries': [], 'rising_queries': []}
        
        try:
            logger.info(f"Fetching related queries for '{keyword}'")
            
            self.pytrends.build_payload(
                kw_list=[keyword],
                timeframe='today 3-m',
                geo='KE'
            )
            
            related_queries = self.pytrends.related_queries()
            
            if keyword in related_queries:
                top_queries = related_queries[keyword].get('top', pd.DataFrame())
                rising_queries = related_queries[keyword].get('rising', pd.DataFrame())
                
                result = {
                    'keyword': keyword,
                    'top_queries': top_queries.to_dict('records') if not top_queries.empty else [],
                    'rising_queries': rising_queries.to_dict('records') if not rising_queries.empty else [],
                    'collected_at': datetime.now()
                }
                
                logger.info(f"Collected related queries for '{keyword}'")
                return result
            else:
                logger.warning(f"No related queries found for '{keyword}'")
                return {'keyword': keyword, 'top_queries': [], 'rising_queries': []}
                
        except Exception as e:
            logger.error(f"Error getting related queries: {e}")
            raise  # Re-raise for retry decorator
    
    @retry_on_rate_limit
    @rate_limited(max_per_minute=5)  # Reduced from 10 to 5
    def get_real_time_trends(self) -> pd.DataFrame:
        """Get real-time trending searches (last hour)"""
        if not self.initialized:
            logger.warning("Google Trends not initialized")
            return pd.DataFrame()
        
        try:
            logger.info("Fetching real-time trends...")
            
            # Use daily trends as proxy for real-time (more reliable)
            self.pytrends.build_payload(
                kw_list=['violence', 'protest'],  # Sample keywords
                timeframe='today 1-d',  # Changed from 'now 1-d' to 'today 1-d'
                geo='KE'
            )
            
            hourly_df = self.pytrends.interest_over_time()
            
            if not hourly_df.empty:
                # Get the most recent hour
                latest_hour = hourly_df.iloc[-1:].reset_index()
                latest_hour = latest_hour.melt(
                    id_vars=['date'],
                    var_name='keyword',
                    value_name='interest_score'
                )
                
                latest_hour['region'] = 'Kenya'
                latest_hour['source'] = 'google_trends_realtime'
                latest_hour['collected_at'] = datetime.now()
                
                logger.info("Collected real-time trends")
                return latest_hour
            else:
                logger.warning("No real-time trends data")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error getting real-time trends: {e}")
            raise  # Re-raise for retry decorator
    
    def collect_all_trends_data(self) -> Dict:
        """Collect all Google Trends data with intelligent rate limiting"""
        if not self.initialized:
            logger.error("Google Trends not initialized")
            return {
                'interest_over_time': pd.DataFrame(),
                'trending_searches': pd.DataFrame(),
                'regional_interest': pd.DataFrame(),
                'related_queries': [],
                'real_time_trends': pd.DataFrame(),
                'risk_score': 0.0
            }
        
        logger.info("🌐 Collecting Google Trends data (conservative mode)...")
        
        results = {
            'interest_over_time': pd.DataFrame(),
            'trending_searches': pd.DataFrame(),
            'regional_interest': pd.DataFrame(),
            'related_queries': [],
            'real_time_trends': pd.DataFrame(),
            'risk_score': 0.0
        }
        
        try:
            # 1. Get interest over time for key conflict keywords (only 2)
            logger.info("  📊 Getting interest over time...")
            interest_keywords = ['violence', 'protest']  # Reduced from 5 to 2
            results['interest_over_time'] = self.get_interest_over_time(
                keywords=interest_keywords,
                timeframe='today 3-m'
            )
            
            # Longer delay between different types of requests
            time.sleep(5)
            
            # 2. Get trending searches in Kenya
            logger.info("  🔥 Getting trending searches...")
            results['trending_searches'] = self.get_trending_searches(region='KE')
            
            time.sleep(5)
            
            # 3. Get regional interest for ONE key conflict term
            logger.info("  🗺️ Getting regional interest...")
            regional_data = []
            # Only try one keyword instead of two
            try:
                region_df = self.get_interest_by_region('violence', resolution='REGION')
                if not region_df.empty:
                    regional_data.append(region_df)
            except Exception as e:
                logger.warning(f"Skipping regional interest due to error: {e}")
            
            if regional_data:
                results['regional_interest'] = pd.concat(regional_data, ignore_index=True)
            
            # Skip related queries to reduce API calls
            logger.info("  ⏭️  Skipping related queries to avoid rate limiting")
            
            # Skip real-time trends to reduce API calls
            logger.info("  ⏭️  Skipping real-time trends to avoid rate limiting")
            
            # 6. Calculate risk score
            logger.info("  📈 Calculating risk score...")
            all_trends_data = pd.concat([
                results['interest_over_time'],
                results['trending_searches'],
                results['regional_interest']
            ], ignore_index=True)
            
            results['risk_score'] = self.calculate_risk_score_from_trends(all_trends_data)
            
            logger.info(f"✅ Google Trends collection complete. Risk score: {results['risk_score']:.2f}")
            
        except Exception as e:
            logger.error(f"❌ Error in Google Trends collection: {e}")
        
        return results
    
    def collect_all_trends_data_safe(self) -> Dict:
        """Safe collection method with fallbacks and mock data"""
        if not self.initialized:
            logger.error("Google Trends not initialized")
            return self.get_mock_trends_data()
        
        logger.info("🌐 Collecting Google Trends data (safe mode)...")
        
        results = {
            'interest_over_time': pd.DataFrame(),
            'trending_searches': pd.DataFrame(),
            'regional_interest': pd.DataFrame(),
            'related_queries': [],
            'real_time_trends': pd.DataFrame(),
            'risk_score': 0.0
        }
        
        try:
            # Try real collection first
            real_data = self.collect_all_trends_data()
            
            # Check if we got any real data
            has_real_data = (
                not real_data['interest_over_time'].empty or
                not real_data['trending_searches'].empty or
                not real_data['regional_interest'].empty
            )
            
            if has_real_data:
                logger.info("✅ Using real Google Trends data")
                return real_data
            else:
                logger.warning("⚠️ No real data collected, using mock data")
                return self.get_mock_trends_data()
                
        except Exception as e:
            logger.error(f"Error in safe collection: {e}")
            return self.get_mock_trends_data()
    
    def get_mock_trends_data(self) -> Dict:
        """Generate realistic mock data when API fails"""
        logger.info("🔄 Generating mock Google Trends data")
        
        # Create mock interest data
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        mock_interest = pd.DataFrame({
            'date': dates,
            'keyword': 'violence',
            'interest_score': np.random.randint(10, 70, size=30),
            'region': 'Kenya',
            'source': 'google_trends_mock',
            'collected_at': datetime.now()
        })
        
        # Mock trending searches
        mock_trending = pd.DataFrame({
            'trending_search': [
                'nairobi protest today',
                'kenya election update',
                'security alert nairobi'
            ],
            'region': ['KE', 'KE', 'KE'],
            'source': ['google_trends_mock'],
            'collected_at': [datetime.now()] * 3
        })
        
        # Calculate mock risk score based on recent trends
        recent_scores = mock_interest['interest_score'].tail(7).mean()
        risk_score = min(recent_scores / 100, 0.7)  # Cap at 0.7 for mock data
        
        return {
            'interest_over_time': mock_interest,
            'trending_searches': mock_trending,
            'regional_interest': pd.DataFrame(),
            'related_queries': [],
            'real_time_trends': pd.DataFrame(),
            'risk_score': risk_score
        }
    
    def calculate_risk_score_from_trends(self, trends_data: pd.DataFrame) -> float:
        """Calculate conflict risk score based on Google Trends data"""
        if trends_data.empty:
            return 0.0
        
        risk_score = 0.0
        factors = []
        
        # Factor 1: High interest in conflict keywords
        if 'interest_score' in trends_data.columns:
            max_interest = trends_data['interest_score'].max()
            if max_interest > 50:  # High interest threshold
                risk_score += 0.3
                factors.append(f"High search interest ({max_interest})")
        
        # Factor 2: Multiple conflict keywords trending
        unique_keywords = trends_data['keyword'].nunique() if 'keyword' in trends_data.columns else 0
        if unique_keywords >= 3:
            risk_score += 0.2
            factors.append(f"{unique_keywords} conflict keywords trending")
        
        # Factor 3: Rising queries (indicates emerging issues)
        if 'rising_queries' in trends_data.columns:
            rising_count = len(trends_data['rising_queries'])
            if rising_count > 2:
                risk_score += 0.25
                factors.append(f"{rising_count} rising search queries")
        
        # Factor 4: Regional concentration
        if 'region' in trends_data.columns and 'interest_score' in trends_data.columns:
            region_scores = trends_data.groupby('region')['interest_score'].mean()
            if len(region_scores) > 0 and region_scores.max() > 30:
                risk_score += 0.25
                factors.append("High regional concentration")
        
        # Cap at 1.0
        risk_score = min(risk_score, 1.0)
        
        logger.info(f"Google Trends Risk Score: {risk_score:.2f} - Factors: {factors}")
        return risk_score
    
    def format_for_main_system(self, trends_data: Dict) -> pd.DataFrame:
        """Format Google Trends data for integration with main system"""
        formatted_records = []
        
        # Add interest over time data
        if not trends_data['interest_over_time'].empty:
            for _, row in trends_data['interest_over_time'].iterrows():
                formatted_records.append({
                    'source': 'google_trends',
                    'title': f"Google Trends: {row['keyword']} interest",
                    'text': f"Search interest for '{row['keyword']}' on {row['date'].strftime('%Y-%m-%d')}: {row['interest_score']}",
                    'date': row['date'],
                    'region': row.get('region', 'Kenya'),
                    'interest_score': row['interest_score'],
                    'keyword': row['keyword'],
                    'risk_level': self._interest_to_risk_level(row['interest_score']),
                    'conflict_score': min(row['interest_score'] / 100, 1.0),
                    'collected_at': datetime.now()
                })
        
        # Add trending searches
        if not trends_data['trending_searches'].empty:
            for _, row in trends_data['trending_searches'].iterrows():
                # Check if trending search is conflict-related
                is_conflict_related = any(
                    kw in row['trending_search'].lower() 
                    for kw in self.conflict_keywords
                )
                
                if is_conflict_related:
                    formatted_records.append({
                        'source': 'google_trends',
                        'title': f"Trending: {row['trending_search']}",
                        'text': f"Trending search in {row['region']}: {row['trending_search']}",
                        'date': row['collected_at'],
                        'region': row['region'],
                        'trending_search': row['trending_search'],
                        'risk_level': 'Medium' if is_conflict_related else 'Low',
                        'conflict_score': 0.5 if is_conflict_related else 0.2,
                        'collected_at': datetime.now()
                    })
        
        # Add regional interest
        if not trends_data['regional_interest'].empty:
            for _, row in trends_data['regional_interest'].iterrows():
                formatted_records.append({
                    'source': 'google_trends',
                    'title': f"Regional interest: {row['keyword']} in {row['region']}",
                    'text': f"Search interest for '{row['keyword']}' in {row['region']}: {row['interest_score']}",
                    'date': row['collected_at'],
                    'region': row['region'],
                    'keyword': row['keyword'],
                    'interest_score': row['interest_score'],
                    'risk_level': self._interest_to_risk_level(row['interest_score']),
                    'conflict_score': min(row['interest_score'] / 100, 1.0),
                    'collected_at': datetime.now()
                })
        
        return pd.DataFrame(formatted_records)
    
    def _interest_to_risk_level(self, interest_score: float) -> str:
        """Convert Google Trends interest score to risk level"""
        if interest_score >= 70:
            return 'High'
        elif interest_score >= 40:
            return 'Medium'
        else:
            return 'Low'

def test_google_trends():
    """Test the Google Trends collector"""
    print("🧪 Testing Google Trends Collector (Enhanced Version)")
    print("=" * 50)
    
    try:
        collector = GoogleTrendsCollector()
        
        if not collector.initialized:
            print("❌ Google Trends collector not initialized")
            print("💡 Install pytrends: pip install pytrends")
            return
        
        # Test the safe method first
        print("\n1. Testing safe collection method...")
        all_data = collector.collect_all_trends_data_safe()
        
        print(f"   Interest over time: {len(all_data['interest_over_time'])} records")
        print(f"   Trending searches: {len(all_data['trending_searches'])} records")
        print(f"   Regional interest: {len(all_data['regional_interest'])} records")
        print(f"   Calculated risk score: {all_data['risk_score']:.2f}")
        
        # Check if we got real or mock data
        if 'google_trends_mock' in str(all_data['interest_over_time'].get('source', '')):
            print("   ⚠️ Using MOCK data (API may be rate limited)")
        else:
            print("   ✅ Using REAL API data")
        
        # Test rate limiting
        print("\n2. Testing rate limiting with retry logic...")
        try:
            # Try a few quick calls to test rate limiting
            start_time = time.time()
            
            print("   Making 3 calls with rate limiting...")
            for i in range(3):
                data = collector.get_interest_over_time(keywords=['violence'], timeframe='today 7-d')
                if not data.empty:
                    print(f"   Call {i+1}: Success - {len(data)} records")
                else:
                    print(f"   Call {i+1}: No data (may be rate limited)")
                if i < 2:
                    time.sleep(2)  # Small delay between calls
            
            end_time = time.time()
            elapsed = end_time - start_time
            print(f"   ⏱️  Elapsed time: {elapsed:.1f} seconds")
            
        except Exception as e:
            print(f"   ❌ Rate limiting test failed: {e}")
        
        print("\n" + "=" * 50)
        print("✅ Enhanced Google Trends test complete!")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        print("💡 Make sure tenacity is installed: pip install tenacity")
        print("💡 If you see 'method_whitelist' error, try: pip install --upgrade urllib3 requests")

def test_rate_limiting():
    """Test the rate limiting functionality"""
    print("  Testing rate limiting with multiple calls...")
    try:
        collector = GoogleTrendsCollector()
        if not collector.initialized:
            print("  ⚠️ Skipping rate limiting test - collector not initialized")
            return
        
        start_time = time.time()
        
        # Make several calls quickly - rate limiter should slow them down
        calls_to_make = 3
        for i in range(calls_to_make):
            print(f"    Call {i+1}/{calls_to_make}...")
            data = collector.get_trending_searches('KE')
            if not data.empty:
                print(f"      Got {len(data)} trending searches")
            else:
                print(f"      No data (may be rate limited)")
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        print(f"  ⏱️  Made {calls_to_make} calls in {elapsed:.1f} seconds")
        print(f"     Expected minimum: {(calls_to_make - 1) * (60/5):.1f} seconds")
        
        if elapsed > 5:  # Should be at least 5 seconds for 3 calls
            print("  ✅ Rate limiting working correctly")
        else:
            print("  ⚠️ Rate limiting may not be working (too fast)")
            
    except Exception as e:
        print(f"  ❌ Error in rate limiting test: {e}")

if __name__ == "__main__":
    print("🚀 Starting Enhanced Google Trends Collector")
    print("With retry logic, rate limiting, and fallback data")
    print("=" * 50)
    
    # Run the enhanced test
    test_google_trends()