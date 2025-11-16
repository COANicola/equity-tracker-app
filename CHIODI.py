"""
COA Equity Tracker - Versione 3.0
Enhanced version with multi-strategy support, annual reports, and improved styling
"""

import streamlit as st
from sqlalchemy import create_engine, Column, Integer, String, Float, Date, Text, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker, scoped_session
import pandas as pd
import yfinance as yf
from passlib.context import CryptContext
import datetime
import plotly.express as px
import plotly.graph_objects as go
import jwt
from jwt.exceptions import PyJWTError
import time
from contextlib import contextmanager
import logging
import io

# ---------- Logging setup ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- Configuration ----------
# Try to get secrets from Streamlit secrets, fallback to environment variables or defaults
try:
    JWT_SECRET = st.secrets.get('jwt_secret', 'dev-secret-change-me-in-production')
    DB_URL = st.secrets.get('db_url', 'sqlite:///equity_app_final.db')
except Exception:
    # Fallback for local development
    import os
    JWT_SECRET = os.environ.get('JWT_SECRET', 'dev-secret-change-me-in-production')
    DB_URL = os.environ.get('DB_URL', 'sqlite:///equity_app_final.db')

JWT_ALGO = 'HS256'
JWT_EXP_SECONDS = 60 * 60 * 8  # 8 hours

# COA Brand Colors based on logo
COA_COLORS = {
    'primary_purple': '#7A2E8F',
    'primary_blue': '#1E8CC8', 
    'light_gray': '#CFCFCF',
    'dark_purple': '#5C1F6B',
    'light_purple': '#9B4FB5',
    'dark_blue': '#166A9B',
    'light_blue': '#4AA8E0',
    'background': '#F8F9FA',
    'card_bg': '#FFFFFF',
    'text_primary': '#2D3748',
    'text_secondary': '#718096',
    'success': '#38A169',
    'warning': '#D69E2E',
    'error': '#E53E3E'
}

# ---------- Custom CSS for COA Branding ----------
def apply_coa_styling():
    # Use session state theme preference
    if hasattr(st.session_state, 'theme'):
        is_dark = st.session_state.theme == 'dark'
    else:
        # Fallback to system detection
        try:
            theme = st.get_option("theme.base")
            is_dark = theme == "dark"
        except:
            is_dark = False
    
    # Adjust colors based on theme
    if is_dark:
        bg_color = "#1a1a1a"
        card_bg = "#2d2d2d"
        text_primary = "#e2e8f0"
        text_secondary = "#a0aec0"
    else:
        bg_color = COA_COLORS['background']
        card_bg = COA_COLORS['card_bg']
        text_primary = COA_COLORS['text_primary']
        text_secondary = COA_COLORS['text_secondary']
    
    st.markdown(f"""
    <style>
    /* Main theme colors - Dynamic based on detected theme */
    :root {{
        --primary-purple: {COA_COLORS['primary_purple']};
        --primary-blue: {COA_COLORS['primary_blue']};
        --light-gray: {COA_COLORS['light_gray']};
        --background: {bg_color};
        --card-bg: {card_bg};
        --text-primary: {text_primary};
        --text-secondary: {text_secondary};
    }}
    
    /* Dark theme support using media query as fallback */
    @media (prefers-color-scheme: dark) {{
        :root {{
            --background: #1a1a1a;
            --card-bg: #2d2d2d;
            --text-primary: #e2e8f0;
            --text-secondary: #a0aec0;
        }}
        
        /* Dark theme specific overrides */
        .stApp {{
            background-color: var(--background);
            color: var(--text-primary);
        }}
        
        .metric-card {{
            background: var(--card-bg);
            border-left: 4px solid var(--primary-purple);
        }}
        
        .stForm {{
            background: var(--card-bg);
            border: 1px solid rgba(122, 46, 143, 0.3);
        }}
        
        .streamlit-expanderHeader {{
            background: var(--card-bg);
            border: 1px solid rgba(122, 46, 143, 0.4);
            color: var(--text-primary);
        }}
        
        /* DataFrame styling for dark theme */
        .dataframe {{
            background-color: var(--card-bg);
            color: var(--text-primary);
        }}
        
        /* Input fields for dark theme */
        .stTextInput > div > div > input {{
            background-color: var(--card-bg);
            color: var(--text-primary);
            border: 1px solid rgba(122, 46, 143, 0.3);
        }}
        
        .stTextArea > div > div > textarea {{
            background-color: var(--card-bg);
            color: var(--text-primary);
            border: 1px solid rgba(122, 46, 143, 0.3);
        }}
        
        /* Select boxes for dark theme */
        .stSelectbox > div > div > div {{
            background-color: var(--card-bg);
            color: var(--text-primary);
            border: 1px solid rgba(122, 46, 143, 0.3);
        }}
        
        /* Date input for dark theme */
        .stDateInput > div > div > input {{
            background-color: var(--card-bg);
            color: var(--text-primary);
            border: 1px solid rgba(122, 46, 143, 0.3);
        }}
        
        /* Number input for dark theme */
        .stNumberInput > div > div > input {{
            background-color: var(--card-bg);
            color: var(--text-primary);
            border: 1px solid rgba(122, 46, 143, 0.3);
        }}
    }}
    
    /* Force light theme for specific elements */
    .main-header {{
        background: linear-gradient(135deg, var(--primary-purple), var(--primary-blue));
        color: white !important;
    }}
    
    .title-container h1 {{
        color: white !important;
    }}
    
    .title-container p {{
        color: rgba(255,255,255,0.9) !important;
    }}
    
    /* Global styles */
    .stApp {{
        background-color: var(--background);
    }}
    
    /* Header styling */
    .main-header {{
        background: linear-gradient(135deg, var(--primary-purple), var(--primary-blue));
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }}
    
    .header-content {{
        display: flex;
        align-items: center;
        gap: 1.5rem;
    }}
    
    .logo-container {{
        background: white;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }}
    
    .title-container h1 {{
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }}
    
    .title-container p {{
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
    }}
    
    /* Card styling */
    .metric-card {{
        background: var(--card-bg);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        border-left: 4px solid var(--primary-purple);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }}
    
    .metric-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(0,0,0,0.12);
    }}
    
    .metric-value {{
        font-size: 2rem;
        font-weight: 700;
        color: var(--primary-purple);
        margin: 0;
    }}
    
    .metric-label {{
        color: var(--text-secondary);
        font-size: 0.9rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin: 0 0 0.5rem 0;
    }}
    
    /* Button styling */
    .stButton > button {{
        background: linear-gradient(135deg, var(--primary-purple), var(--primary-blue));
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(122, 46, 143, 0.3);
    }}
    
    .stButton > button:hover {{
        transform: translateY(-1px);
        box-shadow: 0 4px 16px rgba(122, 46, 143, 0.4);
        background: linear-gradient(135deg, var(--dark-purple), var(--dark-blue));
    }}
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {{
        background: var(--card-bg);
        border-radius: 12px;
        padding: 0.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }}
    
    .stTabs [data-baseweb="tab"] {{
        background: transparent;
        border: none;
        color: var(--text-secondary);
        font-weight: 500;
        padding: 0.75rem 1.25rem;
        border-radius: 8px;
        transition: all 0.2s ease;
    }}
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {{
        background: linear-gradient(135deg, var(--primary-purple), var(--primary-blue));
        color: white;
        font-weight: 600;
    }}
    
    /* Form styling */
    .stForm {{
        background: var(--card-bg);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        border: 1px solid rgba(122, 46, 143, 0.1);
    }}
    
    /* DataFrame styling */
    .dataframe {{
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }}
    
    /* Success/Error messages */
    .stSuccess {{
        background: rgba(56, 161, 105, 0.1);
        border: 1px solid rgba(56, 161, 105, 0.3);
        border-radius: 8px;
        padding: 1rem;
    }}
    
    .stError {{
        background: rgba(229, 62, 62, 0.1);
        border: 1px solid rgba(229, 62, 62, 0.3);
        border-radius: 8px;
        padding: 1rem;
    }}
    
    /* Sidebar styling */
    .css-1d391kg {{  /* Sidebar */
        background: linear-gradient(180deg, var(--primary-purple), var(--primary-blue));
    }}
    
    /* Expander styling */
    .streamlit-expanderHeader {{
        background: var(--card-bg);
        border-radius: 8px;
        border: 1px solid rgba(122, 46, 143, 0.2);
        font-weight: 600;
        color: var(--primary-purple);
    }}
    
    /* Plotly chart styling */
    .js-plotly-plot .plotly .svg-container {{
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
    }}
    </style>
    """, unsafe_allow_html=True)

# ---------- DB setup with multi-strategy support ----------
Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)
    role = Column(String, nullable=False, default='user')
    investor_name = Column(String, nullable=True)

class Strategy(Base):
    __tablename__ = 'strategies'
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)
    description = Column(Text, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(Date, default=datetime.date.today)

class Event(Base):
    __tablename__ = 'events'
    id = Column(Integer, primary_key=True)
    date = Column(Date, nullable=False)
    type = Column(String, nullable=False)  # deposit, withdrawal, valuation
    strategy_id = Column(Integer, nullable=True)  # New field for strategy
    investor = Column(String, nullable=True)
    eur_amount = Column(Float, nullable=True)
    usd_amount = Column(Float, nullable=True)
    eurusd_rate = Column(Float, nullable=True)
    valuation_total_usd = Column(Float, nullable=True)
    note = Column(Text, nullable=True)

if DB_URL.startswith('sqlite'):
    engine = create_engine(DB_URL, connect_args={"check_same_thread": False})
else:
    engine = create_engine(DB_URL)

Base.metadata.create_all(engine)
session_factory = sessionmaker(bind=engine)
Session = scoped_session(session_factory)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# ---------- Context manager for database sessions ----------
@contextmanager
def get_db_session():
    session = Session()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Database error: {e}")
        raise
    finally:
        session.close()

# ---------- Utility functions ----------
def get_historical_eurusd(date: datetime.date, retry_count: int = 3) -> float:
    ticker = 'EURUSD=X'
    for attempt in range(retry_count):
        try:
            start_date = date - datetime.timedelta(days=7)
            end_date = date + datetime.timedelta(days=1)
            data = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), 
                             end=end_date.strftime('%Y-%m-%d'), progress=False)
            if not data.empty:
                data.index = pd.to_datetime(data.index).date
                available_data = data[data.index <= date]
                if not available_data.empty:
                    rate = float(available_data['Close'].iloc[-1])
                    if rate > 0: return rate
        except Exception as e:
            logger.warning(f"yfinance download attempt {attempt + 1} failed: {e}")
            if attempt < retry_count - 1: time.sleep(1)
    logger.error(f"Could not fetch EUR/USD rate for {date}. Using fallback.")
    return 1.10

def hash_password(password: str) -> str: return pwd_context.hash(password)
def verify_password(password: str, password_hash: str) -> bool:
    try: return pwd_context.verify(password, password_hash)
    except Exception: return False

def create_jwt(username: str, role: str) -> str:
    payload = {'sub': username, 'role': role, 'iat': int(time.time()), 'exp': int(time.time()) + JWT_EXP_SECONDS}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGO)

def decode_jwt(token: str):
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGO])
        if payload.get('exp', 0) < int(time.time()): return None
        return payload
    except PyJWTError: return None

# ---------- Enhanced Replay Engine with Strategy Support ----------
def replay_events(events_df: pd.DataFrame, strategy_id: int = None):
    if events_df.empty: return {}, pd.DataFrame()
    
    # Filter by strategy if specified
    if strategy_id is not None:
        events_df = events_df[events_df['strategy_id'] == strategy_id]
    
    events_df = events_df.sort_values('date').reset_index(drop=True)
    investor_balances, history_rows, prev_total_value = {}, [], 0.0
    
    for _, row in events_df.iterrows():
        event_type = row.get('type')
        usd_amount = float(row.get('usd_amount', 0.0) or 0.0)
        investor = row.get('investor')
        
        if event_type == 'deposit' and investor and usd_amount > 0:
            investor_balances[investor] = investor_balances.get(investor, 0.0) + usd_amount
        elif event_type == 'withdrawal' and investor and usd_amount > 0:
            investor_balances[investor] = investor_balances.get(investor, 0.0) - usd_amount
        elif event_type == 'valuation':
            new_total_value = float(row.get('valuation_total_usd', 0.0) or 0.0)
            if new_total_value > 0:
                current_total_from_balances = sum(investor_balances.values())
                if current_total_from_balances > 0:
                    factor = new_total_value / current_total_from_balances
                    for inv in investor_balances: investor_balances[inv] *= factor
                prev_total_value = new_total_value
                snapshot = {'date': row['date'], 'total': prev_total_value, **investor_balances}
                history_rows.append(snapshot)
                continue
        
        prev_total_value = sum(investor_balances.values())
        snapshot = {'date': row['date'], 'total': prev_total_value, **investor_balances}
        history_rows.append(snapshot)
    
    history_df = pd.DataFrame(history_rows) if history_rows else pd.DataFrame()
    if not history_df.empty: history_df = history_df.drop_duplicates(subset='date', keep='last')
    return investor_balances, history_df

# ---------- CSV Export/Import Functions ----------
def export_events_to_csv():
    with get_db_session() as db:
        events_df = pd.read_sql(db.query(Event).statement, db.bind)
        events_df['date'] = pd.to_datetime(events_df['date']).dt.strftime('%Y-%m-%d')
        csv_buffer = io.StringIO()
        events_df.to_csv(csv_buffer, index=False)
        return csv_buffer.getvalue()

def import_events_from_csv(csv_content):
    try:
        df = pd.read_csv(io.StringIO(csv_content))
        required_cols = ['date', 'type', 'strategy_id', 'investor', 'eur_amount', 'usd_amount', 'eurusd_rate', 'valuation_total_usd', 'note']
        
        # Ensure all required columns exist
        for col in required_cols:
            if col not in df.columns:
                df[col] = None
        
        with get_db_session() as db:
            for _, row in df.iterrows():
                event = Event(
                    date=pd.to_datetime(row['date']).date(),
                    type=row['type'],
                    strategy_id=row['strategy_id'] if pd.notna(row['strategy_id']) else None,
                    investor=row['investor'] if pd.notna(row['investor']) else None,
                    eur_amount=float(row['eur_amount']) if pd.notna(row['eur_amount']) else None,
                    usd_amount=float(row['usd_amount']) if pd.notna(row['usd_amount']) else None,
                    eurusd_rate=float(row['eurusd_rate']) if pd.notna(row['eurusd_rate']) else None,
                    valuation_total_usd=float(row['valuation_total_usd']) if pd.notna(row['valuation_total_usd']) else None,
                    note=row['note'] if pd.notna(row['note']) else None
                )
                db.add(event)
        return True, f"Successfully imported {len(df)} events"
    except Exception as e:
        return False, f"Import failed: {str(e)}"

# ---------- Annual Report Functions ----------
def generate_annual_report(year: int, strategy_id: int = None):
    with get_db_session() as db:
        start_date = datetime.date(year, 1, 1)
        end_date = datetime.date(year, 12, 31)
        query = db.query(Event).filter(Event.date.between(start_date, end_date))
        if strategy_id:
            query = query.filter(Event.strategy_id == strategy_id)
        
        events_df = pd.read_sql(query.statement, db.bind)
        events_df['date'] = pd.to_datetime(events_df['date']).dt.date
        
        if events_df.empty:
            return None, None
        
        # Calculate monthly performance
        monthly_data = []
        for month in range(1, 13):
            month_start = datetime.date(year, month, 1)
            if month == 12:
                month_end = datetime.date(year, 12, 31)
            else:
                month_end = datetime.date(year, month + 1, 1) - datetime.timedelta(days=1)
            
            month_events = events_df[events_df['date'].between(month_start, month_end)]
            
            deposits = month_events[month_events['type'] == 'deposit']['usd_amount'].sum()
            withdrawals = month_events[month_events['type'] == 'withdrawal']['usd_amount'].sum()
            
            # Get last valuation of the month
            month_valuations = month_events[month_events['type'] == 'valuation']
            end_value = month_valuations['valuation_total_usd'].iloc[-1] if not month_valuations.empty else None
            
            monthly_data.append({
                'Month': datetime.date(year, month, 1).strftime('%B'),
                'Deposits': deposits,
                'Withdrawals': withdrawals,
                'Net_Flow': deposits - withdrawals,
                'End_Value': end_value
            })
        
        monthly_df = pd.DataFrame(monthly_data)
        
        # Calculate annual metrics
        total_deposits = events_df[events_df['type'] == 'deposit']['usd_amount'].sum()
        total_withdrawals = events_df[events_df['type'] == 'withdrawal']['usd_amount'].sum()
        
        # Get start and end values
        start_valuations = events_df[events_df['date'] >= datetime.date(year, 1, 1)]
        start_valuation_rows = start_valuations[start_valuations['type'] == 'valuation']['valuation_total_usd']
        start_value = start_valuation_rows.iloc[0] if not start_valuation_rows.empty else None
        
        end_valuations = events_df[events_df['date'] <= datetime.date(year, 12, 31)]
        end_valuation_rows = end_valuations[end_valuations['type'] == 'valuation']['valuation_total_usd']
        end_value = end_valuation_rows.iloc[-1] if not end_valuation_rows.empty else None
        
        annual_metrics = {
            'total_deposits': total_deposits,
            'total_withdrawals': total_withdrawals,
            'net_investment': total_deposits - total_withdrawals,
            'start_value': start_value,
            'end_value': end_value,
            'total_return': (end_value - start_value) if start_value and end_value else None,
            'return_percentage': ((end_value - start_value) / start_value * 100) if start_value and end_value and start_value > 0 else None
        }
        
        return monthly_df, annual_metrics

# ---------- Main App ----------
st.set_page_config(page_title='COA Equity Tracker', page_icon='📊', layout='wide')
apply_coa_styling()

# Initialize session state
if 'jwt' not in st.session_state:
    st.session_state.jwt, st.session_state.username, st.session_state.role = None, None, None

# Initialize theme preference
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'  # Default to light theme

# ---------- Authentication ----------
with get_db_session() as db:
    if db.query(User).count() == 0:
        u = User(username='admin', password_hash=hash_password('admin123'), role='admin')
        db.add(u)
        st.info('Default admin user created. Login with admin / admin123')

if not st.session_state.jwt:
    # Login page with COA branding
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # COA Logo and Branding
        try:
            st.image("COA_no sfondo_no scritta.png", width=90)
        except Exception as e:
            logger.info(f"Logo display failed: {e}")
            # Fallback to text logo with COA branding
            st.markdown(f"""
            <div style="text-align: center; margin-bottom: 2rem;">
                <div style="background: linear-gradient(135deg, {COA_COLORS['primary_purple']}, {COA_COLORS['primary_blue']});
                            color: white; padding: 2rem; border-radius: 16px; text-align: center;">
                    <h1 style="margin: 0; font-weight: 700; font-size: 3rem;">COA-Portfolio</h1>
                    <p style="margin: 0.5rem 0 0 0; opacity: 0.9; font-size: 1.1rem;">Gestione e analisi del portafoglio COA</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<h2 style='text-align: center; margin-bottom: 1.5rem;'>COA-Portfolio</h2>", unsafe_allow_html=True)
        
        with st.form("login_form"):
            login_user = st.text_input('Username', placeholder='Enter your username')
            login_pw = st.text_input('Password', type='password', placeholder='Enter your password')
            
            col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
            with col_btn2:
                if st.form_submit_button('Login', use_container_width=True):
                    with get_db_session() as db:
                        user = db.query(User).filter(User.username == login_user).first()
                        if user and verify_password(login_pw, user.password_hash):
                            st.session_state.jwt, st.session_state.username, st.session_state.role = create_jwt(user.username, user.role), user.username, user.role
                            st.rerun()
                        else: 
                            st.error('❌ Invalid credentials')
        
        # Professional footer
        st.markdown("""
        <div style="text-align: center; margin-top: 2rem; opacity: 0.7;">
            <p style="font-size: 0.9rem;">Gestione e analisi del portafoglio COA</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.stop()

# Verify JWT
payload = decode_jwt(st.session_state.jwt)
if not payload:
    st.error('⏰ Session invalid or expired. Please re-login.')
    st.session_state.clear()
    if st.button('🔙 Back to Login'): st.rerun()
    st.stop()

current_user, current_role = st.session_state.username, st.session_state.role

# ---------- Main App Header ----------
col1, col2, col3 = st.columns([1, 4, 1])
with col1:
    # Create a container to center the logo vertically
    with st.container():
        # Add some vertical spacing to center the logo
        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
        try:
            # Try to display logo if it exists - use the actual filename
            st.image("COA_no sfondo_no scritta.png", width=110)
        except Exception as e:
            logger.info(f"Logo display failed: {e}")
            # Fallback to text logo
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, {COA_COLORS['primary_purple']}, {COA_COLORS['primary_blue']});
                        color: white; padding: 1rem; border-radius: 12px; text-align: center;">
                <h2 style="margin: 0; font-weight: 700;">COA</h2>
            </div>
            """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="main-header">
        <div class="header-content">
            <div class="title-container">
                <h1>COA Equity Tracker</h1>
                <p>Gestione e analisi del portafoglio COA</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    info_col, logout_col = st.columns([3, 1])
    with info_col:
        st.markdown(f"""
        <div style="background: white; padding: 1rem; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); text-align: center;">
            <div style="color: {COA_COLORS['primary_purple']}; font-weight: 600; margin-bottom: 0.5rem;">
                👤 {current_user}
            </div>
            <div style="color: {COA_COLORS['text_secondary']}; font-size: 0.9rem; margin-bottom: 1rem;">
                {current_role.title()}
            </div>
            {f'<div style="color: {COA_COLORS["success"]}; font-size: 0.8rem;">✓ Active</div>' if payload else ''}
        </div>
        """, unsafe_allow_html=True)
    with logout_col:
        if st.button('Esci', use_container_width=True, key='logout_btn'):
            st.session_state.clear()
            st.rerun()

st.divider()

# ---------- Data Loading ----------
with get_db_session() as db:
    user_obj = db.query(User).filter(User.username == current_user).first()
    user_investor_name = user_obj.investor_name if user_obj else None
    
    # Load all events
    all_events_df = pd.read_sql(db.query(Event).statement, db.bind)
    all_events_df['date'] = pd.to_datetime(all_events_df['date']).dt.date
    
    # Load strategies
    strategies = db.query(Strategy).filter(Strategy.is_active == True).all()
    strategies_df = pd.read_sql(db.query(Strategy).statement, db.bind)

# ---------- Protocol Management (Admin Only) ----------
if current_role == 'admin':
    with st.expander('🎯 Protocol Management'):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader('➕ Add New Protocol')
            with st.form("add_protocol_form"):
                protocol_name = st.text_input('Protocol Name', placeholder='e.g., Growth Protocol')
                protocol_desc = st.text_area('Description', placeholder='Describe the protocol...')
                
                if st.form_submit_button('Add Protocol', use_container_width=True):
                    if protocol_name:
                        with get_db_session() as db:
                            existing = db.query(Strategy).filter(Strategy.name == protocol_name).first()
                            if existing:
                                st.error('Protocol name already exists')
                            else:
                                new_protocol = Strategy(name=protocol_name, description=protocol_desc)
                                db.add(new_protocol)
                                db.commit()  # Commit the transaction
                                st.success(f'Protocol "{protocol_name}" added successfully!')
                                time.sleep(1)
                                st.rerun()
                    else:
                        st.error('Protocol name is required')
        
        with col2:
            st.subheader('📋 Active Protocols')
            # Reload protocols within session context to avoid DetachedInstanceError
            with get_db_session() as db:
                active_protocols = db.query(Strategy).filter(Strategy.is_active == True).all()
                if active_protocols:
                    for protocol in active_protocols:
                        with st.container():
                            col_a, col_b, col_c = st.columns([3, 1, 1])
                            with col_a:
                                st.markdown(f"**{protocol.name}**")
                                if protocol.description:
                                    st.caption(protocol.description)
                            with col_b:
                                if st.button('✏️', key=f'rename_protocol_{protocol.id}', help='Rename Protocol'):
                                    st.session_state[f'renaming_protocol_{protocol.id}'] = True
                                    st.session_state[f'rename_name_{protocol.id}'] = protocol.name
                                    st.session_state[f'rename_desc_{protocol.id}'] = protocol.description or ''
                            with col_c:
                                if st.button('🗑️', key=f'del_protocol_{protocol.id}', help='Delete Protocol'):
                                    protocol_obj = db.get(Strategy, protocol.id)
                                    protocol_obj.is_active = False
                                    db.commit()
                                    st.success('Protocol deactivated')
                                    time.sleep(1)
                                    st.rerun()
                            
                            # Rename form (shown when rename button is clicked)
                            if st.session_state.get(f'renaming_protocol_{protocol.id}', False):
                                with st.form(f'rename_form_{protocol.id}'):
                                    new_name = st.text_input('New Name', 
                                                             value=st.session_state[f'rename_name_{protocol.id}'],
                                                             key=f'rename_name_input_{protocol.id}')
                                    new_desc = st.text_area('New Description (optional)', 
                                                            value=st.session_state[f'rename_desc_{protocol.id}'],
                                                            key=f'rename_desc_input_{protocol.id}')
                                    
                                    col_rename1, col_rename2 = st.columns(2)
                                    with col_rename1:
                                        if st.form_submit_button('💾 Save', use_container_width=True):
                                            if new_name and new_name != protocol.name:
                                                # Check if name already exists
                                                existing = db.query(Strategy).filter(Strategy.name == new_name).first()
                                                if existing:
                                                    st.error('Protocol name already exists')
                                                else:
                                                    protocol_obj = db.get(Strategy, protocol.id)
                                                    protocol_obj.name = new_name
                                                    protocol_obj.description = new_desc
                                                    db.commit()
                                                    st.success(f'Protocol renamed to "{new_name}"')
                                                    # Clear rename state
                                                    st.session_state[f'renaming_protocol_{protocol.id}'] = False
                                                    time.sleep(1)
                                                    st.rerun()
                                            elif new_name == protocol.name:
                                                # Only update description if name is the same
                                                protocol_obj = db.get(Strategy, protocol.id)
                                                protocol_obj.description = new_desc
                                                db.commit()
                                                st.success('Protocol description updated')
                                                st.session_state[f'renaming_protocol_{protocol.id}'] = False
                                                time.sleep(1)
                                                st.rerun()
                                    with col_rename2:
                                        if st.form_submit_button('❌ Cancel', use_container_width=True):
                                            st.session_state[f'renaming_protocol_{protocol.id}'] = False
                                            st.rerun()
                else:
                    st.info('No protocols defined yet')

# ---------- CSV Export/Import (Admin Only) ----------
if current_role == 'admin':
    with st.expander('📁 Data Management'):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader('📤 Export Data')
            if st.button('Export All Events to CSV', use_container_width=True):
                csv_data = export_events_to_csv()
                st.download_button(
                    label="⬇️ Download CSV",
                    data=csv_data,
                    file_name=f"coa_events_export_{datetime.date.today().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        with col2:
            st.subheader('📥 Import Data')
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
            if uploaded_file is not None:
                csv_content = uploaded_file.getvalue().decode('utf-8')
                success, message = import_events_from_csv(csv_content)
                if success:
                    st.success(message)
                    time.sleep(2)
                    st.rerun()
                else:
                    st.error(message)

# ---------- Main Dashboard ----------
if all_events_df.empty:
    st.info('📊 No events in the system yet. Start by adding your first deposit!')
    
    if current_role == 'admin':
        with st.form("first_deposit_form"):
            st.subheader("➕ Add First Event")
            col1, col2 = st.columns(2)
            
            with col1:
                d_date = st.date_input('Date', datetime.date.today())
                investor = st.text_input('Investor Name', placeholder='Enter investor name')
                
            with col2:
                # Protocol selection - reload within session to avoid DetachedInstanceError
                with get_db_session() as db:
                    current_protocols = db.query(Strategy).filter(Strategy.is_active == True).all()
                    protocol_options = ['No Protocol'] + [s.name for s in current_protocols]
                    selected_protocol = st.selectbox('Protocol', protocol_options)
                
            eur_amount = st.number_input('Amount (EUR)', min_value=0.01, step=100.0, value=1000.0)
            
            if st.form_submit_button('💰 Add First Deposit', use_container_width=True):
                rate = get_historical_eurusd(d_date)
                usd_amount = eur_amount * rate
                
                with get_db_session() as db:
                    strategy_id = None
                    if selected_protocol != 'No Protocol':
                        protocol = db.query(Strategy).filter(Strategy.name == selected_protocol).first()
                        strategy_id = protocol.id if protocol else None
                    
                    db.add(Event(
                        date=d_date, 
                        type='deposit', 
                        strategy_id=strategy_id,
                        investor=investor.strip(), 
                        eur_amount=eur_amount, 
                        usd_amount=usd_amount, 
                        eurusd_rate=rate
                    ))
                
                st.success('🎉 First deposit added successfully!')
                time.sleep(2)
                st.rerun()
    else:
        st.stop()

else:
    # Filter events based on user role and strategy selection
    selected_strategy_id = st.session_state.get('selected_strategy_id', None)
    
    if current_role != 'admin':
        if user_investor_name:
            events_df = all_events_df[
                (all_events_df['type'] == 'valuation') | 
                (all_events_df['investor'] == user_investor_name)
            ]
        else:
            events_df = all_events_df[all_events_df['type'] == 'valuation']
    else:
        events_df = all_events_df

    # Calculate balances and history for main portfolio
    total_balances, total_history = replay_events(all_events_df)
    total_portfolio_value = sum(total_balances.values())
    
    # Calculate strategy-specific data
    strategy_data = {}
    # Reload strategies within session context to avoid DetachedInstanceError
    with get_db_session() as db:
        active_strategies = db.query(Strategy).filter(Strategy.is_active == True).all()
        for strategy in active_strategies:
            strat_balances, strat_history = replay_events(all_events_df, strategy.id)
            strategy_data[strategy.name] = {
                'balances': strat_balances,
                'history': strat_history,
                'total_value': sum(strat_balances.values())
            }
    
    # Calculate overall ROI
    overall_roi = 0
    if total_portfolio_value > 0:
        total_usd_invested_overall = all_events_df[all_events_df['type'] == 'deposit']['usd_amount'].sum()
        if total_usd_invested_overall > 0:
            overall_roi = ((total_portfolio_value - total_usd_invested_overall) / total_usd_invested_overall * 100)

    # ---------- Key Metrics ----------
    st.markdown("### 📊 Portfolio Overview")
    
    # Main metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Portfolio Value</div>
            <div class="metric-value">${total_portfolio_value:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Overall ROI</div>
            <div class="metric-value" style="color: {'#38A169' if overall_roi >= 0 else '#E53E3E'}">
                {overall_roi:+.2f}%
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        if current_role != 'admin' and user_investor_name:
            user_balance = total_balances.get(user_investor_name, 0.0)
            user_share = (user_balance / total_portfolio_value * 100) if total_portfolio_value > 0 else 0
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Your Share</div>
                <div class="metric-value">{user_share:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            visible_investors = sorted({inv for inv in events_df['investor'].dropna().unique()})
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Active Investors</div>
                <div class="metric-value">{len(visible_investors)}</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col4:
        # Reload strategies within session to avoid DetachedInstanceError
        with get_db_session() as db:
            current_strategies = db.query(Strategy).filter(Strategy.is_active == True).all()
            if current_strategies:
                active_strategies = len([s for s in current_strategies if s.is_active])
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Active Strategies</div>
                    <div class="metric-value">{active_strategies}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Strategies</div>
                    <div class="metric-value">None</div>
                </div>
                """, unsafe_allow_html=True)

    st.divider()

    # ---------- Navigation Tabs ----------
    tab_list = ["📈 Dashboard", "👥 Investor Details"]
    if current_role == 'admin':
        tab_list.extend(["⚙️ Event Management"])
    
    tabs = st.tabs(tab_list)

    # Dashboard Tab
    with tabs[0]:
        st.markdown("### 📈 Portfolio Performance")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Main portfolio chart
            if not total_history.empty:
                fig_portfolio = px.line(
                    total_history, 
                    x='date', 
                    y='total', 
                    title='Total Portfolio Value Over Time',
                    labels={'date': 'Date', 'total': 'Portfolio Value (USD)'},
                    line_shape='linear',
                    render_mode='svg'
                )
                fig_portfolio.update_traces(
                    line_color=COA_COLORS['primary_purple'],
                    line_width=3,
                    marker_size=6,
                    marker_color=COA_COLORS['primary_blue']
                )
                fig_portfolio.update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(color=COA_COLORS['text_primary']),
                    title_font_size=16,
                    title_font_color=COA_COLORS['primary_purple'],
                    height=400
                )
                st.plotly_chart(fig_portfolio, use_container_width=True)
            else:
                st.info("No historical data available yet")
        
        with col2:
            # Strategy performance comparison
            if strategy_data:
                strategy_names = list(strategy_data.keys())
                strategy_values = [data['total_value'] for data in strategy_data.values()]
                
                fig_strategies = px.pie(
                    names=strategy_names,
                    values=strategy_values,
                    title='Strategy Allocation',
                    color_discrete_sequence=[COA_COLORS['primary_purple'], COA_COLORS['primary_blue'], 
                                           COA_COLORS['light_purple'], COA_COLORS['light_blue']]
                )
                fig_strategies.update_layout(height=400)
                st.plotly_chart(fig_strategies, use_container_width=True)
        
        # Add shared charts like in CHIODI_old.py
        if current_role == 'admin':
            st.subheader("📈 Individual Share Values (USD) Over Time")
            if not total_history.empty and len(total_history.columns) > 2:
                investor_cols = [c for c in total_history.columns if c not in ['date', 'total']]
                if investor_cols:
                    value_df_melted = total_history.melt(id_vars='date', value_vars=investor_cols, var_name='Investor', value_name='USD Value')
                    fig_investor_value = px.line(value_df_melted, x='date', y='USD Value', color='Investor', markers=True, 
                                               labels={'USD Value': 'Share Value (USD)', 'date': 'Date'})
                    st.plotly_chart(fig_investor_value, use_container_width=True)
        else: 
            st.subheader("📈 Your Value Trend (USD)")
            user_investor_name = None
            with get_db_session() as db:
                user = db.query(User).filter(User.username == current_user).first()
                if user and user.investor_name:
                    user_investor_name = user.investor_name
            
            if user_investor_name and user_investor_name in total_history.columns:
                user_history_df = total_history[['date', user_investor_name]].copy()
                user_history_df.rename(columns={user_investor_name: 'Your Value (USD)'}, inplace=True)
                fig_user_value = px.line(user_history_df, x='date', y='Your Value (USD)', markers=True)
                st.plotly_chart(fig_user_value, use_container_width=True)
        
        st.subheader("📊 Investor Shares Evolution (%)")
        history_to_show = total_history.copy()
        if not history_to_show.empty and len(history_to_show.columns) > 2:
            inv_cols = [c for c in history_to_show.columns if c not in ['date', 'total']]
            if inv_cols:
                pct_df = history_to_show[inv_cols].div(history_to_show['total'], axis=0).fillna(0)
                pct_df['date'] = history_to_show['date']
                pct_df_melted = pct_df.melt(id_vars='date', var_name='Investor', value_name='Share')
                fig_shares = px.area(pct_df_melted, x='date', y='Share', color='Investor', markers=True)
                fig_shares.update_yaxes(tickformat='.0%')
                st.plotly_chart(fig_shares, use_container_width=True)

    # Investor Details Tab
    with tabs[1]:
        st.markdown("### 👥 Investor Performance")
        
        investors_to_show = sorted({inv for inv in events_df['investor'].dropna().unique()})
        investor_data = []
        
        for inv in investors_to_show:
            inv_deposits = all_events_df[(all_events_df['investor'] == inv) & (all_events_df['type'] == 'deposit')]
            inv_withdrawals = all_events_df[(all_events_df['investor'] == inv) & (all_events_df['type'] == 'withdrawal')]
            
            total_usd_invested = inv_deposits['usd_amount'].sum()
            total_withdrawn_usd = inv_withdrawals['usd_amount'].sum()
            current_usd_value = total_balances.get(inv, 0.0)
            
            roi_pct = ((current_usd_value + total_withdrawn_usd - total_usd_invested) / total_usd_invested * 100) if total_usd_invested > 0 else 0
            
            investor_data.append({
                'Investor': inv,
                'EUR Invested': inv_deposits['eur_amount'].sum(),
                'USD Invested': total_usd_invested,
                'Total Withdrawn (USD)': total_withdrawn_usd,
                'Current Value (USD)': current_usd_value,
                'Share %': (current_usd_value / total_portfolio_value * 100) if total_portfolio_value > 0 else 0,
                'ROI %': roi_pct
            })
        
        if investor_data:
            df_inv = pd.DataFrame(investor_data)
            
            # Color coding for ROI
            def roi_color(val):
                if pd.isna(val) or val == 0:
                    return f'background-color: #FEF3C7; color: {COA_COLORS["text_primary"]};'
                elif val > 0:
                    intensity = min(abs(val) / 50, 1)  # Normalize to 0-1
                    green_intensity = int(255 * (1 - intensity * 0.5))
                    return f'background-color: rgb({int(255 * (1 - intensity))}, {green_intensity}, {int(255 * (1 - intensity))}); color: white; font-weight: 600;'
                else:
                    intensity = min(abs(val) / 50, 1)  # Normalize to 0-1
                    red_intensity = int(255 * (1 - intensity * 0.5))
                    return f'background-color: rgb({red_intensity}, {int(255 * (1 - intensity))}, {int(255 * (1 - intensity))}); color: white; font-weight: 600;'
            
            styled_df = df_inv.style.format({
                'EUR Invested': '€{:,.2f}',
                'USD Invested': '${:,.2f}',
                'Total Withdrawn (USD)': '${:,.2f}',
                'Current Value (USD)': '${:,.2f}',
                'Share %': '{:.2f}%',
                'ROI %': '{:+.2f}%'
            }).apply(lambda x: x.map(roi_color), subset=['ROI %'])
            
            st.dataframe(styled_df, use_container_width=True, hide_index=True)

    # Event Management Tab (Admin Only)
    if current_role == 'admin':
        with tabs[2] if len(tabs) > 2 else st.container():
            st.markdown("### ⚙️ Event Management")
            
            form_cols = st.columns(2)
            
            with form_cols[0]:
                st.subheader("➕ Add Event")
                
                tab_dep, tab_wd, tab_val = st.tabs(["💰 Deposit", "💸 Withdrawal", "📈 Valuation"])
                
                with tab_dep:
                    with st.form("deposit_form"):
                        d_date = st.date_input('Date', datetime.date.today(), key='d_date')
                        d_investor = st.text_input('Investor Name', placeholder='Enter investor name', key='d_inv')
                        
                        # Reload protocols within session to avoid DetachedInstanceError
                        with get_db_session() as db:
                            current_protocols = db.query(Strategy).filter(Strategy.is_active == True).all()
                            protocol_options = ['No Protocol'] + [s.name for s in current_protocols]
                            d_protocol = st.selectbox('Protocol', protocol_options, key='d_protocol')
                        
                        d_eur = st.number_input('Amount (EUR)', min_value=0.01, step=100.0, key='d_eur')
                        
                        if st.form_submit_button('💰 Add Deposit', use_container_width=True):
                            rate = get_historical_eurusd(d_date)
                            usd_amount = d_eur * rate
                            
                            with get_db_session() as db:
                                strategy_id = None
                                if d_protocol != 'No Protocol':
                                    protocol = db.query(Strategy).filter(Strategy.name == d_protocol).first()
                                    strategy_id = protocol.id if protocol else None
                                
                                db.add(Event(
                                    date=d_date, 
                                    type='deposit', 
                                    strategy_id=strategy_id,
                                    investor=d_investor.strip(), 
                                    eur_amount=d_eur, 
                                    usd_amount=usd_amount, 
                                    eurusd_rate=rate
                                ))
                            
                            st.success('Deposit added successfully!')
                            time.sleep(1)
                            st.rerun()
                
                with tab_wd:
                    with st.form("withdrawal_form"):
                        w_date = st.date_input('Date', datetime.date.today(), key='w_date')
                        w_investor = st.text_input('Investor Name', placeholder='Enter investor name', key='w_inv')
                        
                        # Reload strategies within session to avoid DetachedInstanceError
                        with get_db_session() as db:
                            current_protocols = db.query(Strategy).filter(Strategy.is_active == True).all()
                            protocol_options = ['No Protocol'] + [s.name for s in current_protocols]
                            w_protocol = st.selectbox('Protocol', protocol_options, key='w_protocol')
                        
                        w_usd = st.number_input('Amount (USD)', min_value=0.01, step=100.0, key='w_usd')
                        
                        if st.form_submit_button('💸 Add Withdrawal', use_container_width=True):
                            rate = get_historical_eurusd(w_date)
                            eur_amount = w_usd / rate
                            
                            with get_db_session() as db:
                                strategy_id = None
                                if w_protocol != 'No Protocol':
                                    protocol = db.query(Strategy).filter(Strategy.name == w_protocol).first()
                                    strategy_id = protocol.id if protocol else None
                                
                                db.add(Event(
                                    date=w_date, 
                                    type='withdrawal', 
                                    strategy_id=strategy_id,
                                    investor=w_investor.strip(), 
                                    eur_amount=eur_amount, 
                                    usd_amount=w_usd, 
                                    eurusd_rate=rate
                                ))
                            
                            st.success('Withdrawal added successfully!')
                            time.sleep(1)
                            st.rerun()
                
                with tab_val:
                    with st.form("valuation_form"):
                        v_date = st.date_input('Date', datetime.date.today(), key='v_date')
                        
                        # Reload protocols within session to avoid DetachedInstanceError
                        with get_db_session() as db:
                            current_protocols = db.query(Strategy).filter(Strategy.is_active == True).all()
                            protocol_options = ['All Protocols'] + [s.name for s in current_protocols]
                            v_protocol = st.selectbox('Protocol', protocol_options, key='v_protocol')
                        
                        v_total = st.number_input('Total Portfolio Value (USD)', min_value=0.01, step=1000.0, key='v_usd')
                        
                        if st.form_submit_button('📈 Add Valuation', use_container_width=True):
                            with get_db_session() as db:
                                strategy_id = None
                                if v_protocol != 'All Protocols':
                                    protocol = db.query(Strategy).filter(Strategy.name == v_protocol).first()
                                    strategy_id = protocol.id if protocol else None
                                
                                db.add(Event(
                                    date=v_date, 
                                    type='valuation', 
                                    strategy_id=strategy_id,
                                    valuation_total_usd=v_total
                                ))
                            
                            st.success('Valuation added successfully!')
                            time.sleep(1)
                            st.rerun()
            
            with form_cols[1]:
                st.subheader("✏️ Edit/Delete Events")
                
                with get_db_session() as db:
                    events_to_edit = pd.read_sql(db.query(Event).statement, db.bind)
                    
                    if not events_to_edit.empty:
                        # Add strategy names
                        strategy_mapping = strategies_df[['id', 'name']].set_index('id')['name'].to_dict()
                        events_to_edit['protocol_name'] = events_to_edit['strategy_id'].map(strategy_mapping).fillna('No Protocol')
                        
                        st.dataframe(
                            events_to_edit[['id', 'date', 'type', 'protocol_name', 'investor', 'eur_amount', 'usd_amount', 'valuation_total_usd']].sort_values('date', ascending=False), 
                            height=300, 
                            use_container_width=True, 
                            hide_index=True
                        )
                        
                        ev_id_to_edit = st.selectbox('Select Event ID to edit', options=events_to_edit['id'].tolist())
                        
                        if ev_id_to_edit:
                            ev_row = db.get(Event, ev_id_to_edit)
                            if ev_row:
                                # Load protocols within the same session to avoid DetachedInstanceError
                                current_protocols = db.query(Strategy).filter(Strategy.is_active == True).all()
                                protocol_mapping = {s.name: s for s in current_protocols}
                                
                                with st.form(f"edit_form_{ev_row.id}"):
                                    st.markdown(f"**Edit Event #{ev_row.id} ({ev_row.type})**")
                                    
                                    e_date = st.date_input('Date', value=ev_row.date)
                                    
                                    if ev_row.type in ['deposit', 'withdrawal']:
                                        e_investor = st.text_input('Investor Name', value=ev_row.investor or "")
                                        
                                        current_protocol = 'No Protocol'
                                        if ev_row.strategy_id:
                                            protocol = db.query(Strategy).filter(Strategy.id == ev_row.strategy_id).first()
                                            current_protocol = protocol.name if protocol else 'No Protocol'
                                        
                                        protocol_options = ['No Protocol'] + [s.name for s in current_protocols]
                                        e_protocol = st.selectbox('Protocol', protocol_options, 
                                                                index=protocol_options.index(current_protocol))
                                    
                                    if ev_row.type == 'deposit':
                                        e_eur = st.number_input('Amount (EUR)', value=ev_row.eur_amount or 0.0)
                                    elif ev_row.type == 'withdrawal':
                                        e_usd = st.number_input('Amount (USD)', value=ev_row.usd_amount or 0.0)
                                    else:  # valuation
                                        e_val = st.number_input('Total Value (USD)', value=ev_row.valuation_total_usd or 0.0)
                                    
                                    col_btn1, col_btn2 = st.columns(2)
                                    
                                    with col_btn1:
                                        if st.form_submit_button('💾 Update', use_container_width=True):
                                            ev_row.date = e_date
                                            rate = get_historical_eurusd(e_date)
                                            
                                            if ev_row.type in ['deposit', 'withdrawal']:
                                                ev_row.investor = e_investor.strip() or None
                                                
                                                # Update protocol
                                                if e_protocol != 'No Protocol':
                                                    protocol = protocol_mapping.get(e_protocol)
                                                    ev_row.strategy_id = protocol.id if protocol else None
                                                else:
                                                    ev_row.strategy_id = None
                                            
                                            if ev_row.type == 'deposit':
                                                ev_row.eurusd_rate = rate
                                                ev_row.usd_amount = e_eur * rate
                                                ev_row.eur_amount = e_eur
                                            elif ev_row.type == 'withdrawal':
                                                ev_row.eurusd_rate = rate
                                                ev_row.eur_amount = e_usd / rate
                                                ev_row.usd_amount = e_usd
                                            else:  # valuation
                                                ev_row.valuation_total_usd = e_val
                                            
                                            db.commit()
                                            st.success(f"Event #{ev_row.id} updated!")
                                            time.sleep(1)
                                            st.rerun()
                                    
                                    with col_btn2:
                                        if st.form_submit_button('🗑️ Delete', type="primary", use_container_width=True):
                                            db.delete(ev_row)
                                            db.commit()
                                            st.success(f"Event #{ev_row.id} deleted!")
                                            time.sleep(1)
                                            st.rerun()
                    else:
                        st.info("No events to edit")

# ---------- Admin Panel ----------
if current_role == 'admin':
    with st.expander('👑 Admin Panel: User Management'):
        admin_cols = st.columns(2)
        
        with admin_cols[0]:
            st.subheader('➕ Create New User')
            with st.form("create_user_form"):
                new_username = st.text_input('Username', placeholder='Enter username')
                new_password = st.text_input('Password', type='password', placeholder='Enter password')
                new_role = st.selectbox('Role', ['user', 'admin'])
                new_investor_name = st.text_input('Investor Name (optional)', placeholder='Link to investor')
                
                if st.form_submit_button('Create User', use_container_width=True):
                    if new_username and new_password:
                        try:
                            with get_db_session() as db:
                                if db.query(User).filter(User.username == new_username).first():
                                    st.error('Username already exists')
                                else:
                                    db.add(User(
                                        username=new_username.strip(), 
                                        password_hash=hash_password(new_password), 
                                        role=new_role, 
                                        investor_name=new_investor_name.strip() or None
                                    ))
                                    db.commit()  # Commit the transaction
                                    st.success(f'User "{new_username}" created successfully!')
                                    time.sleep(1)
                                    st.rerun()
                        except Exception as e:
                            st.error(f"Database error: {e}")
                    else:
                        st.warning('Username and password are required')
        
        with admin_cols[1]:
            st.subheader('✏️ Edit/Delete User')
            with get_db_session() as db:
                all_users = db.query(User).all()
                
                if all_users:
                    user_options = [f"{user.username} ({user.role})" for user in all_users]
                    selected_user = st.selectbox("Select user", options=user_options)
                    
                    if selected_user:
                        username = selected_user.split(' (')[0]
                        user_to_edit = db.query(User).filter(User.username == username).first()
                        
                        if user_to_edit:
                            with st.form("edit_user_form"):
                                is_admin_user = (user_to_edit.username == 'admin')
                                
                                st.markdown(f"**Managing User: {user_to_edit.username}**")
                                
                                edited_username = st.text_input(
                                    "Username", 
                                    value=user_to_edit.username, 
                                    disabled=is_admin_user
                                )
                                
                                edited_role = st.selectbox(
                                    "Role", 
                                    ['user', 'admin'], 
                                    index=1 if user_to_edit.role == 'admin' else 0, 
                                    disabled=is_admin_user
                                )
                                
                                edited_inv_name = st.text_input(
                                    "Investor Name", 
                                    value=user_to_edit.investor_name or "",
                                    disabled=is_admin_user
                                )
                                
                                new_password = st.text_input(
                                    "New Password (leave blank to keep current)", 
                                    type="password"
                                )
                                
                                col_btn1, col_btn2 = st.columns(2)
                                
                                with col_btn1:
                                    if st.form_submit_button("💾 Update User", use_container_width=True):
                                        if not is_admin_user:
                                            user_to_edit.investor_name = edited_inv_name.strip() or None
                                            
                                            if edited_username != user_to_edit.username:
                                                existing = db.query(User).filter(User.username == edited_username).first()
                                                if existing:
                                                    st.error("Username already in use")
                                                else:
                                                    user_to_edit.username = edited_username
                                            
                                            user_to_edit.role = edited_role
                                        
                                        if new_password:
                                            user_to_edit.password_hash = hash_password(new_password)
                                        
                                        db.commit()
                                        st.success(f"User '{user_to_edit.username}' updated!")
                                        time.sleep(1)
                                        st.rerun()
                                
                                with col_btn2:
                                    if not is_admin_user and st.form_submit_button(
                                        "🗑️ DELETE USER", 
                                        type="primary", 
                                        use_container_width=True
                                    ):
                                        db.delete(user_to_edit)
                                        db.commit()
                                        st.warning(f"User '{user_to_edit.username}' deleted!")
                                        time.sleep(1)
                                        st.rerun()
                else:
                    st.info("No users to edit")
        
        st.divider()
        st.subheader('👥 Current Users')
        with get_db_session() as db:
            users_df = pd.read_sql(db.query(User).statement, db.bind)
            if not users_df.empty:
                # Remove password hashes for security
                display_df = users_df[['id', 'username', 'role', 'investor_name']].copy()
                st.dataframe(display_df, use_container_width=True, hide_index=True)
            else:
                st.info("No users found")