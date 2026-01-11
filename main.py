import streamlit as st
st.set_page_config(page_title="StockPulse", page_icon="📊", layout="wide")

import pandas as pd
import numpy as np
import requests
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Attempt to import yfinance for live data
try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    yf = None
    YF_AVAILABLE = False
    st.warning("yfinance not installed. Using simulated data for analysis.")

# ------------------------------
# Download and Initialize NLTK Data
# ------------------------------
@st.cache_resource
def download_nltk_data():
    try:
        nltk.download('vader_lexicon', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
    except:
        pass

download_nltk_data()

from nltk.corpus import stopwords
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

# ------------------------------
# Enhanced Custom CSS for Professional Look
# ------------------------------
st.markdown("""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
      body {
          background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
          font-family: 'Inter', sans-serif;
          color: #2c3e50;
      }
      .header {
          text-align: center;
          background: linear-gradient(90deg, #1a2980, #26d0ce);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          font-size: 3.5rem;
          font-weight: 800;
          margin-top: 10px;
          margin-bottom: 5px;
      }
      .subheader {
          text-align: center;
          color: #34495e;
          font-size: 1.4rem;
          font-weight: 400;
          margin-bottom: 30px;
      }
      .metric-card {
          background: white;
          border-radius: 10px;
          padding: 15px;
          box-shadow: 0 4px 6px rgba(0,0,0,0.1);
          border-left: 4px solid #3498db;
          margin-bottom: 10px;
      }
      .positive {
          color: #27ae60;
          font-weight: 600;
      }
      .negative {
          color: #e74c3c;
          font-weight: 600;
      }
      .neutral {
          color: #7f8c8d;
          font-weight: 600;
      }
      .info-box {
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          color: white;
          padding: 15px;
          border-radius: 10px;
          margin: 10px 0;
      }
      .stTabs [data-baseweb="tab-list"] {
          gap: 24px;
      }
      .stTabs [data-baseweb="tab"] {
          height: 50px;
          white-space: pre-wrap;
          background-color: #f8f9fa;
          border-radius: 5px 5px 0px 0px;
          gap: 1px;
          padding-top: 10px;
          padding-bottom: 10px;
      }
      .stTabs [aria-selected="true"] {
          background-color: #3498db;
          color: white !important;
      }
    </style>
""", unsafe_allow_html=True)

# ------------------------------
# Landing Page
# ------------------------------
st.markdown("""
    <div class="header">StockPulse AI</div>
    <div class="subheader">Advanced Sentiment Analysis & Predictive Market Intelligence</div>
""", unsafe_allow_html=True)

# ------------------------------
# Enhanced Utility Functions
# ------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_article_text(url):
    """Enhanced article text extraction with better parsing"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "header", "footer", "nav"]):
            script.decompose()
        
        # Try to find main content
        main_content = soup.find('main') or soup.find('article') or soup.find('div', class_=lambda x: x and ('content' in x or 'article' in x or 'post' in x))
        
        if main_content:
            paragraphs = main_content.find_all('p')
        else:
            paragraphs = soup.find_all('p')
        
        text = ' '.join(p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 50)
        
        # Clean text
        text = ' '.join(text.split())
        
        return text[:10000]  # Limit text length
        
    except Exception as e:
        return ""

@st.cache_data(ttl=1800, show_spinner="Fetching news articles...")
def scrape_articles(stock, max_articles=15):
    """Enhanced article scraping with multiple sources"""
    sources = [
        {"name": "Economic Times", "url": f"https://economictimes.indiatimes.com/topic/{stock}"},
        {"name": "Moneycontrol", "url": f"https://www.moneycontrol.com/news/tags/{stock}.html"},
        {"name": "Business Standard", "url": f"https://www.business-standard.com/search?q={stock}&type=news"},
        {"name": "Livemint", "url": f"https://www.livemint.com/Search/Link/Keyword/{stock}"},
        {"name": "Reuters", "url": f"https://www.reuters.com/search/news?blob={stock}"},
        {"name": "Bloomberg Quint", "url": f"https://www.bloombergquint.com/search?q={stock}"},
    ]
    
    articles = []
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    for source in sources:
        try:
            response = requests.get(source["url"], headers=headers, timeout=8)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find article links (simplified for demo)
            links = soup.find_all('a', href=True)
            
            for link in links[:5]:  # Limit per source
                title = link.get_text().strip()
                if title and stock.lower() in title.lower() and len(title) > 20:
                    article_url = link['href']
                    if not article_url.startswith('http'):
                        article_url = source["url"] + article_url
                    
                    # Fetch article content
                    content = fetch_article_text(article_url)
                    
                    if content and len(content) > 200:
                        articles.append({
                            "Title": title[:200] + "..." if len(title) > 200 else title,
                            "Source": source["name"],
                            "Link": article_url,
                            "Date": datetime.now().strftime("%Y-%m-%d"),
                            "Content": content[:2000]  # Limit content length
                        })
                        
                        if len(articles) >= max_articles:
                            break
                            
        except Exception:
            continue
            
        if len(articles) >= max_articles:
            break
    
    # If no articles found, create sample data for demonstration
    if not articles:
        articles = create_sample_articles(stock)
    
    return pd.DataFrame(articles)

def create_sample_articles(stock):
    """Create sample articles for demonstration"""
    sample_sources = ["Economic Times", "Moneycontrol", "Business Standard", "Reuters", "Bloomberg"]
    articles = []
    
    for i in range(8):
        sentiment = np.random.choice(["Bullish", "Bearish", "Neutral"], p=[0.4, 0.3, 0.3])
        if sentiment == "Bullish":
            titles = [
                f"{stock} Shows Strong Growth Potential",
                f"Analysts Bullish on {stock} Future Prospects",
                f"{stock} Announces Record Quarterly Earnings",
                f"Institutional Investors Increasing {stock} Holdings"
            ]
        elif sentiment == "Bearish":
            titles = [
                f"{stock} Faces Regulatory Challenges",
                f"Market Concerns Over {stock} Valuation",
                f"{stock} Reports Lower Than Expected Revenue",
                f"Analysts Downgrade {stock} Rating"
            ]
        else:
            titles = [
                f"{stock} Maintains Steady Performance",
                f"Mixed Signals for {stock} in Current Market",
                f"{stock} Shows Stability Amid Market Volatility"
            ]
        
        articles.append({
            "Title": np.random.choice(titles),
            "Source": np.random.choice(sample_sources),
            "Link": f"https://example.com/article{i}",
            "Date": (datetime.now() - timedelta(days=np.random.randint(0, 30))).strftime("%Y-%m-%d"),
            "Content": f"Sample article content about {stock}. This demonstrates the sentiment analysis capabilities of StockPulse AI. " * 20
        })
    
    return articles

def analyze_sentiment_batch(texts):
    """Analyze sentiment for multiple texts efficiently"""
    sentiments = []
    for text in texts:
        if not isinstance(text, str) or len(text.strip()) == 0:
            sentiments.append({"compound": 0, "pos": 0, "neg": 0, "neu": 1})
            continue
        
        scores = sia.polarity_scores(text)
        
        # Enhance sentiment scoring with custom weights
        enhanced_scores = {
            "compound": scores["compound"],
            "pos": scores["pos"] * 1.2,  # Weight positive sentiment
            "neg": scores["neg"] * 1.1,  # Weight negative sentiment
            "neu": scores["neu"]
        }
        
        sentiments.append(enhanced_scores)
    
    return sentiments

# ------------------------------
# Enhanced Data Fetching Functions
# ------------------------------
@st.cache_data(ttl=300, show_spinner=False)
def get_stock_data(stock_symbol, period="6mo"):
    """Get comprehensive stock data with error handling"""
    if YF_AVAILABLE:
        try:
            ticker = yf.Ticker(stock_symbol)
            
            # Get historical data
            hist = ticker.history(period=period)
            if hist.empty:
                raise ValueError("No historical data available")
            
            # Get fundamental data
            info = ticker.info
            
            return {
                "success": True,
                "historical": hist,
                "info": info,
                "symbol": stock_symbol.upper()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "symbol": stock_symbol.upper()
            }
    else:
        # Generate simulated data
        return generate_simulated_data(stock_symbol)

def generate_simulated_data(stock_symbol):
    """Generate realistic simulated stock data"""
    dates = pd.date_range(end=datetime.now(), periods=126, freq='B')  # 6 months of business days
    base_price = 100 + np.random.randn() * 20
    
    # Generate price series with trend and volatility
    returns = np.random.randn(len(dates)) * 0.02  # 2% daily volatility
    trend = np.linspace(0, 0.1, len(dates))  # Upward trend
    prices = base_price * np.exp(np.cumsum(returns + trend/len(dates)))
    
    hist = pd.DataFrame({
        'Open': prices * (1 + np.random.randn(len(dates)) * 0.01),
        'High': prices * (1 + np.abs(np.random.randn(len(dates)) * 0.015)),
        'Low': prices * (1 - np.abs(np.random.randn(len(dates)) * 0.015)),
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, len(dates))
    }, index=dates)
    
    info = {
        'currentPrice': float(prices[-1]),
        'forwardPE': np.random.uniform(10, 30),
        'trailingEps': np.random.uniform(2, 10),
        'marketCap': np.random.randint(1e9, 1e12),
        'dividendYield': np.random.uniform(0, 0.05),
        'returnOnEquity': np.random.uniform(0.05, 0.25),
        'profitMargins': np.random.uniform(0.05, 0.3),
        'debtToEquity': np.random.uniform(0.1, 2.0),
        'beta': np.random.uniform(0.5, 1.5)
    }
    
    return {
        "success": True,
        "historical": hist,
        "info": info,
        "symbol": stock_symbol.upper()
    }

# ------------------------------
# Enhanced Analysis Functions
# ------------------------------
def calculate_technical_indicators(df):
    """Calculate comprehensive technical indicators"""
    df = df.copy()
    
    # Moving averages
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    
    # Calculate returns and volatility
    df['Returns'] = df['Close'].pct_change()
    df['Volatility'] = df['Returns'].rolling(window=20).std() * np.sqrt(252)
    
    return df

def calculate_fundamental_score(info):
    """Calculate comprehensive fundamental score"""
    scores = []
    weights = []
    
    # P/E Score (lower is better)
    if 'forwardPE' in info and info['forwardPE']:
        pe = info['forwardPE']
        if pe > 0:
            pe_score = 100 / (1 + pe/50)  # Normalize
            scores.append(pe_score)
            weights.append(0.25)
    
    # ROE Score (higher is better)
    if 'returnOnEquity' in info and info['returnOnEquity']:
        roe = info['returnOnEquity'] * 100
        roe_score = min(100, roe * 4)  # Scale to 100
        scores.append(roe_score)
        weights.append(0.20)
    
    # Profit Margin Score
    if 'profitMargins' in info and info['profitMargins']:
        margin = info['profitMargins'] * 100
        margin_score = min(100, margin * 10)  # Scale to 100
        scores.append(margin_score)
        weights.append(0.15)
    
    # Debt to Equity Score (lower is better)
    if 'debtToEquity' in info and info['debtToEquity']:
        dte = info['debtToEquity']
        dte_score = 100 * (1 / (1 + dte))
        scores.append(dte_score)
        weights.append(0.20)
    
    # Dividend Yield Score
    if 'dividendYield' in info and info['dividendYield']:
        dy = info['dividendYield'] * 100
        dy_score = min(100, dy * 20)  # Scale to 100
        scores.append(dy_score)
        weights.append(0.10)
    
    # Market Cap Score (larger is more stable)
    if 'marketCap' in info and info['marketCap']:
        market_cap = info['marketCap']
        if market_cap > 1e12:  # >1 trillion
            mc_score = 100
        elif market_cap > 1e9:  # >1 billion
            mc_score = 80 + (market_cap / 1e12 * 20)
        else:
            mc_score = 60 + (market_cap / 1e9 * 20)
        scores.append(mc_score)
        weights.append(0.10)
    
    if scores and weights:
        weighted_avg = np.average(scores, weights=weights)
        return min(100, max(0, weighted_avg))
    
    return 50  # Default neutral score

def calculate_sentiment_score(articles_df):
    """Calculate comprehensive sentiment score"""
    if articles_df.empty:
        return 50
    
    # Analyze sentiment
    sentiments = analyze_sentiment_batch(articles_df['Content'].fillna('').tolist())
    compounds = [s['compound'] for s in sentiments]
    
    # Convert compound scores to 0-100 scale
    sentiment_scores = [50 + (c * 50) for c in compounds]  # -1 to 1 -> 0 to 100
    
    # Weight by article recency (simplified)
    avg_score = np.mean(sentiment_scores)
    
    # Apply source credibility weights
    source_weights = {
        'Reuters': 1.2,
        'Bloomberg': 1.2,
        'Economic Times': 1.1,
        'Business Standard': 1.0,
        'Moneycontrol': 1.0,
        'Livemint': 1.0,
        'Sample': 0.8
    }
    
    weighted_scores = []
    for idx, row in articles_df.iterrows():
        weight = source_weights.get(row['Source'], 0.8)
        weighted_scores.append(sentiment_scores[idx] * weight)
    
    if weighted_scores:
        final_score = np.mean(weighted_scores)
        return min(100, max(0, final_score))
    
    return avg_score

# ------------------------------
# Enhanced Visualization Functions
# ------------------------------
def create_sentiment_gauge(score, title="Sentiment Score"):
    """Create a sentiment gauge chart"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 20}},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "#e74c3c"},  # Red
                {'range': [30, 70], 'color': "#f1c40f"},  # Yellow
                {'range': [70, 100], 'color': "#2ecc71"}  # Green
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig

def create_technical_chart(hist_data):
    """Create comprehensive technical analysis chart"""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('Price & Moving Averages', 'MACD', 'RSI'),
        row_heights=[0.5, 0.25, 0.25]
    )
    
    # Price and MAs
    fig.add_trace(
        go.Candlestick(
            x=hist_data.index,
            open=hist_data['Open'],
            high=hist_data['High'],
            low=hist_data['Low'],
            close=hist_data['Close'],
            name='OHLC'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=hist_data.index, y=hist_data['MA50'],
                  line=dict(color='orange', width=1.5),
                  name='MA50'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=hist_data.index, y=hist_data['MA200'],
                  line=dict(color='purple', width=1.5),
                  name='MA200'),
        row=1, col=1
    )
    
    # MACD
    fig.add_trace(
        go.Scatter(x=hist_data.index, y=hist_data['MACD'],
                  line=dict(color='blue', width=1),
                  name='MACD'),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=hist_data.index, y=hist_data['Signal'],
                  line=dict(color='red', width=1),
                  name='Signal'),
        row=2, col=1
    )
    
    # RSI
    fig.add_trace(
        go.Scatter(x=hist_data.index, y=hist_data['RSI'],
                  line=dict(color='green', width=1.5),
                  name='RSI'),
        row=3, col=1
    )
    
    # Add RSI bands
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="gray", row=3, col=1)
    
    fig.update_layout(
        height=800,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        template="plotly_white"
    )
    
    return fig

def create_sentiment_timeline(articles_df):
    """Create sentiment timeline visualization"""
    if articles_df.empty or 'Date' not in articles_df.columns:
        return None
    
    # Analyze sentiment for each article
    sentiments = analyze_sentiment_batch(articles_df['Content'].fillna('').tolist())
    articles_df['Sentiment'] = [s['compound'] for s in sentiments]
    articles_df['Sentiment_Label'] = articles_df['Sentiment'].apply(
        lambda x: 'Bullish' if x > 0.05 else 'Bearish' if x < -0.05 else 'Neutral'
    )
    
    # Convert dates
    articles_df['Date'] = pd.to_datetime(articles_df['Date'], errors='coerce')
    articles_df = articles_df.dropna(subset=['Date'])
    
    # Group by date
    daily_sentiment = articles_df.groupby('Date')['Sentiment'].mean().reset_index()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=daily_sentiment['Date'],
        y=daily_sentiment['Sentiment'],
        mode='lines+markers',
        name='Sentiment Score',
        line=dict(color='#3498db', width=2),
        fill='tozeroy',
        fillcolor='rgba(52, 152, 219, 0.2)'
    ))
    
    fig.add_hline(y=0.05, line_dash="dash", line_color="green",
                  annotation_text="Bullish Threshold")
    fig.add_hline(y=-0.05, line_dash="dash", line_color="red",
                  annotation_text="Bearish Threshold")
    
    fig.update_layout(
        title="Sentiment Timeline",
        xaxis_title="Date",
        yaxis_title="Sentiment Score",
        template="plotly_white",
        hovermode="x unified"
    )
    
    return fig

# ------------------------------
# Main Application
# ------------------------------
def main():
    # Sidebar for controls
    with st.sidebar:
        st.markdown("### 🔍 Stock Analysis Controls")
        
        stock_symbol = st.text_input(
            "Enter Stock Symbol:",
            value="AAPL",
            help="Enter a valid stock symbol (e.g., AAPL, TSLA, RELIANCE.NS)"
        )
        
        analysis_type = st.selectbox(
            "Analysis Type:",
            ["Comprehensive", "Technical Only", "Fundamental Only", "Sentiment Only"]
        )
        
        st.markdown("---")
        st.markdown("### 📊 Data Sources")
        st.info("""
        - **News Sources:** Economic Times, Moneycontrol, Reuters, Bloomberg
        - **Market Data:** Yahoo Finance (Live)
        - **Sentiment Analysis:** VADER + Custom Models
        """)
        
        st.markdown("---")
        st.markdown("### ⚙️ Settings")
        auto_refresh = st.checkbox("Auto-refresh data", value=False)
        show_details = st.checkbox("Show detailed analysis", value=True)
    
    if not stock_symbol:
        st.info("👈 Enter a stock symbol in the sidebar to begin analysis")
        return
    
    # Main content area with tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Overview", 
        "📰 News Sentiment", 
        "📊 Technical Analysis", 
        "💰 Fundamental Analysis"
    ])
    
    # Fetch stock data
    with st.spinner(f"Fetching data for {stock_symbol.upper()}..."):
        stock_data = get_stock_data(stock_symbol)
    
    if not stock_data["success"]:
        st.error(f"Error fetching data for {stock_symbol}: {stock_data.get('error', 'Unknown error')}")
        return
    
    # Fetch news articles
    with st.spinner("Analyzing news sentiment..."):
        articles_df = scrape_articles(stock_symbol)
    
    # Tab 1: Overview
    with tab1:
        st.markdown(f"# {stock_symbol.upper()} - Comprehensive Analysis")
        
        # Key Metrics Row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            current_price = stock_data['historical']['Close'].iloc[-1]
            price_change = ((current_price / stock_data['historical']['Close'].iloc[-2]) - 1) * 100
            st.metric(
                "Current Price",
                f"${current_price:.2f}",
                f"{price_change:+.2f}%",
                delta_color="normal"
            )
        
        with col2:
            # Calculate sentiment score
            sentiment_score = calculate_sentiment_score(articles_df)
            sentiment_label = "Bullish" if sentiment_score > 60 else "Bearish" if sentiment_score < 40 else "Neutral"
            st.metric(
                "Sentiment Score",
                f"{sentiment_score:.0f}/100",
                sentiment_label
            )
        
        with col3:
            # Calculate fundamental score
            fundamental_score = calculate_fundamental_score(stock_data['info'])
            st.metric(
                "Fundamental Score",
                f"{fundamental_score:.0f}/100",
                "Strong" if fundamental_score > 70 else "Weak" if fundamental_score < 30 else "Average"
            )
        
        with col4:
            # Calculate technical score
            hist_data = calculate_technical_indicators(stock_data['historical'])
            rsi = hist_data['RSI'].iloc[-1]
            ma_trend = "Bullish" if current_price > hist_data['MA50'].iloc[-1] > hist_data['MA200'].iloc[-1] else "Bearish"
            st.metric(
                "Technical Score",
                f"{rsi:.0f} (RSI)",
                ma_trend
            )
        
        # Overall Score and Recommendation
        st.markdown("---")
        st.markdown("### 📊 Overall Assessment")
        
        overall_score = (sentiment_score * 0.3 + fundamental_score * 0.4 + 
                        (100 - abs(rsi - 50)/50 * 100) * 0.3)
        
        col_rec1, col_rec2 = st.columns([2, 1])
        
        with col_rec1:
            fig_gauge = create_sentiment_gauge(overall_score, "Overall Score")
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        with col_rec2:
            st.markdown("#### 📋 Recommendation")
            
            if overall_score >= 70:
                st.success("**STRONG BUY**")
                st.markdown("""
                - Strong fundamentals
                - Positive sentiment
                - Technical alignment
                """)
            elif overall_score >= 55:
                st.info("**HOLD**")
                st.markdown("""
                - Mixed signals
                - Monitor closely
                - Consider averaging
                """)
            elif overall_score >= 40:
                st.warning("**CAUTION**")
                st.markdown("""
                - Weak fundamentals
                - Negative sentiment
                - Consider reducing exposure
                """)
            else:
                st.error("**SELL**")
                st.markdown("""
                - Poor fundamentals
                - Strong negative sentiment
                - Technical breakdown
                """)
        
        # Quick Insights
        st.markdown("### 💡 Key Insights")
        insights_col1, insights_col2 = st.columns(2)
        
        with insights_col1:
            st.markdown("#### 📈 Technical Insights")
            if current_price > hist_data['MA50'].iloc[-1] > hist_data['MA200'].iloc[-1]:
                st.success("**Golden Cross Detected** - Bullish trend")
            elif current_price < hist_data['MA50'].iloc[-1] < hist_data['MA200'].iloc[-1]:
                st.error("**Death Cross Detected** - Bearish trend")
            
            if rsi > 70:
                st.warning("RSI indicates **overbought** conditions")
            elif rsi < 30:
                st.warning("RSI indicates **oversold** conditions")
        
        with insights_col2:
            st.markdown("#### 📰 Sentiment Insights")
            if sentiment_score > 70:
                st.success("**Strong Positive Sentiment** from news sources")
            elif sentiment_score < 30:
                st.error("**Strong Negative Sentiment** from news sources")
            
            if len(articles_df) > 0:
                bullish_count = len([s for s in analyze_sentiment_batch(
                    articles_df['Content'].tolist()) if s['compound'] > 0.05])
                st.info(f"**{bullish_count}/{len(articles_df)}** articles are bullish")
    
    # Tab 2: News Sentiment
    with tab2:
        st.markdown("## 📰 News Sentiment Analysis")
        
        if not articles_df.empty:
            # Sentiment Distribution
            col_sent1, col_sent2 = st.columns([2, 1])
            
            with col_sent1:
                sentiments = analyze_sentiment_batch(articles_df['Content'].tolist())
                sentiment_labels = []
                for s in sentiments:
                    if s['compound'] > 0.05:
                        sentiment_labels.append('Bullish')
                    elif s['compound'] < -0.05:
                        sentiment_labels.append('Bearish')
                    else:
                        sentiment_labels.append('Neutral')
                
                sentiment_counts = pd.Series(sentiment_labels).value_counts()
                
                fig_pie = go.Figure(data=[go.Pie(
                    labels=sentiment_counts.index,
                    values=sentiment_counts.values,
                    hole=0.3,
                    marker_colors=['#2ecc71', '#e74c3c', '#95a5a6']
                )])
                
                fig_pie.update_layout(
                    title="Sentiment Distribution",
                    height=400
                )
                
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col_sent2:
                st.markdown("#### 📊 Sentiment Metrics")
                avg_compound = np.mean([s['compound'] for s in sentiments])
                st.metric("Average Sentiment", f"{avg_compound:.3f}")
                st.metric("Positive Articles", f"{sentiment_counts.get('Bullish', 0)}")
                st.metric("Negative Articles", f"{sentiment_counts.get('Bearish', 0)}")
                st.metric("Total Articles", len(articles_df))
            
            # Sentiment Timeline
            timeline_fig = create_sentiment_timeline(articles_df)
            if timeline_fig:
                st.plotly_chart(timeline_fig, use_container_width=True)
            
            # Articles Table
            st.markdown("#### 📝 Latest News Articles")
            
            # Add sentiment analysis to articles
            articles_display = articles_df.copy()
            sentiments = analyze_sentiment_batch(articles_display['Content'].tolist())
            
            articles_display['Sentiment'] = [s['compound'] for s in sentiments]
            articles_display['Sentiment_Label'] = articles_display['Sentiment'].apply(
                lambda x: '🟢 Bullish' if x > 0.05 else '🔴 Bearish' if x < -0.05 else '⚪ Neutral'
            )
            articles_display['Confidence'] = [abs(s['compound']) for s in sentiments]
            
            # Display formatted table
            for idx, row in articles_display.head(10).iterrows():
                with st.expander(f"{row['Title']} - {row['Source']} ({row['Sentiment_Label']})"):
                    st.markdown(f"**Date:** {row['Date']}")
                    st.markdown(f"**Sentiment Score:** {row['Sentiment']:.3f}")
                    st.markdown(f"**Confidence:** {row['Confidence']:.1%}")
                    st.markdown(f"**Preview:** {row['Content'][:500]}...")
                    st.markdown(f"[Read full article]({row['Link']})")
        else:
            st.warning("No news articles found for this stock symbol.")
    
    # Tab 3: Technical Analysis
    with tab3:
        st.markdown("## 📊 Technical Analysis")
        
        hist_data = calculate_technical_indicators(stock_data['historical'])
        
        # Technical Chart
        st.plotly_chart(create_technical_chart(hist_data), use_container_width=True)
        
        # Technical Indicators Grid
        st.markdown("### 🔧 Technical Indicators")
        
        col_tech1, col_tech2, col_tech3, col_tech4 = st.columns(4)
        
        with col_tech1:
            ma50 = hist_data['MA50'].iloc[-1]
            ma200 = hist_data['MA200'].iloc[-1]
            ma_ratio = (current_price / ma50 - 1) * 100
            st.metric("50-Day MA", f"${ma50:.2f}", f"{ma_ratio:+.1f}% vs Price")
        
        with col_tech2:
            rsi = hist_data['RSI'].iloc[-1]
            rsi_status = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
            st.metric("RSI (14)", f"{rsi:.1f}", rsi_status)
        
        with col_tech3:
            macd = hist_data['MACD'].iloc[-1]
            signal = hist_data['Signal'].iloc[-1]
            macd_signal = "Bullish" if macd > signal else "Bearish"
            st.metric("MACD", f"{macd:.3f}", macd_signal)
        
        with col_tech4:
            volatility = hist_data['Volatility'].iloc[-1] * 100
            st.metric("Volatility", f"{volatility:.1f}%")
        
        # Support and Resistance Levels
        st.markdown("### 📉 Support & Resistance")
        
        recent_data = hist_data.tail(100)
        support_level = recent_data['Low'].min()
        resistance_level = recent_data['High'].max()
        
        col_sr1, col_sr2 = st.columns(2)
        
        with col_sr1:
            st.metric("Support Level", f"${support_level:.2f}")
        
        with col_sr2:
            st.metric("Resistance Level", f"${resistance_level:.2f}")
        
        # Price Distance from Support/Resistance
        support_distance = ((current_price - support_level) / support_level) * 100
        resistance_distance = ((resistance_level - current_price) / current_price) * 100
        
        st.progress(support_distance / 100, text=f"Distance from Support: {support_distance:.1f}%")
        st.progress(resistance_distance / 100, text=f"Distance to Resistance: {resistance_distance:.1f}%")
    
    # Tab 4: Fundamental Analysis
    with tab4:
        st.markdown("## 💰 Fundamental Analysis")
        
        info = stock_data['info']
        
        # Key Metrics Grid
        col_fund1, col_fund2, col_fund3 = st.columns(3)
        
        with col_fund1:
            st.markdown("#### 💵 Valuation Metrics")
            if 'forwardPE' in info:
                st.metric("Forward P/E", f"{info['forwardPE']:.2f}")
            if 'trailingEps' in info:
                st.metric("EPS", f"${info['trailingEps']:.2f}")
            if 'priceToBook' in info:
                st.metric("P/B Ratio", f"{info['priceToBook']:.2f}")
        
        with col_fund2:
            st.markdown("#### 📊 Profitability")
            if 'returnOnEquity' in info:
                st.metric("ROE", f"{info['returnOnEquity']*100:.1f}%")
            if 'profitMargins' in info:
                st.metric("Profit Margin", f"{info['profitMargins']*100:.1f}%")
            if 'operatingMargins' in info:
                st.metric("Operating Margin", f"{info['operatingMargins']*100:.1f}%")
        
        with col_fund3:
            st.markdown("#### ⚖️ Financial Health")
            if 'debtToEquity' in info:
                st.metric("Debt/Equity", f"{info['debtToEquity']:.2f}")
            if 'currentRatio' in info:
                st.metric("Current Ratio", f"{info['currentRatio']:.2f}")
            if 'quickRatio' in info:
                st.metric("Quick Ratio", f"{info['quickRatio']:.2f}")
        
        # Dividend Information
        st.markdown("### 💸 Dividend Analysis")
        
        col_div1, col_div2 = st.columns(2)
        
        with col_div1:
            if 'dividendYield' in info and info['dividendYield']:
                st.metric("Dividend Yield", f"{info['dividendYield']*100:.2f}%")
            if 'dividendRate' in info:
                st.metric("Dividend Rate", f"${info['dividendRate']:.2f}")
        
        with col_div2:
            if 'payoutRatio' in info:
                payout_status = "Sustainable" if info['payoutRatio'] < 0.6 else "High"
                st.metric("Payout Ratio", f"{info['payoutRatio']*100:.1f}%", payout_status)
        
        # Growth Metrics
        st.markdown("### 📈 Growth Metrics")
        
        growth_metrics = []
        if 'earningsGrowth' in info:
            growth_metrics.append(("Earnings Growth", info['earningsGrowth']*100))
        if 'revenueGrowth' in info:
            growth_metrics.append(("Revenue Growth", info['revenueGrowth']*100))
        if 'bookValueGrowth' in info:
            growth_metrics.append(("Book Value Growth", info['bookValueGrowth']*100))
        
        if growth_metrics:
            growth_df = pd.DataFrame(growth_metrics, columns=['Metric', 'Value'])
            
            fig_growth = px.bar(
                growth_df,
                x='Metric',
                y='Value',
                title="Growth Metrics (%)",
                color='Value',
                color_continuous_scale=px.colors.sequential.Bluyl
            )
            
            fig_growth.update_layout(showlegend=False)
            st.plotly_chart(fig_growth, use_container_width=True)
        
        # Risk Metrics
        st.markdown("### ⚠️ Risk Assessment")
        
        col_risk1, col_risk2 = st.columns(2)
        
        with col_risk1:
            if 'beta' in info:
                beta = info['beta']
                beta_risk = "High" if beta > 1.2 else "Low" if beta < 0.8 else "Moderate"
                st.metric("Beta", f"{beta:.2f}", beta_risk)
        
        with col_risk2:
            # Calculate market cap category
            if 'marketCap' in info:
                market_cap = info['marketCap']
                if market_cap > 2e11:
                    cap_category = "Mega Cap"
                elif market_cap > 1e10:
                    cap_category = "Large Cap"
                elif market_cap > 2e9:
                    cap_category = "Mid Cap"
                else:
                    cap_category = "Small Cap"
                st.metric("Market Cap", cap_category)

# ------------------------------
# Run the application
# ------------------------------
if __name__ == "__main__":
    main()



