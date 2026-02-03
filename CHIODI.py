"""
COA Equity Tracker - Versione 3.0
Enhanced version with multi-strategy support, annual reports, and improved styling
"""

import streamlit as st
from sqlalchemy import create_engine, Column, Integer, String, Float, Date, Text, Boolean, func
from sqlalchemy.orm import declarative_base, sessionmaker, scoped_session
import pandas as pd
import yfinance as yf
from passlib.context import CryptContext
import datetime
import calendar
import plotly.express as px
import plotly.graph_objects as go
import jwt
from jwt.exceptions import PyJWTError
import time
from contextlib import contextmanager
import logging
import io
import base64
from PIL import Image

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

def load_coa_logo(path: str = "COA_no sfondo_no scritta.png"):
    try:
        img = Image.open(path)
        if img.mode in ("RGBA", "LA"):
            alpha = img.split()[-1]
            bbox = alpha.getbbox()
            if bbox:
                img = img.crop(bbox)
        return img
    except Exception as e:
        logger.info(f"Logo load failed: {e}")
        return None

def render_logo_centered(path: str = "COA_no sfondo_no scritta.png", width_px: int = 100):
    try:
        img = load_coa_logo(path)
        if img is not None:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            data = buf.getvalue()
        else:
            with open(path, "rb") as f:
                data = f.read()
        b64 = base64.b64encode(data).decode("ascii")
        st.markdown(
            f"<div style='display:flex;justify-content:center'><img src='data:image/png;base64,{b64}' style='width:{width_px}px'/></div>",
            unsafe_allow_html=True,
        )
    except Exception as e:
        try:
            st.image(path, width=width_px)
        except Exception:
            logger.info(f"Logo render failed: {e}")

def get_logo_img_tag(path: str = "COA_no sfondo_no scritta.png", width_px: int = 140) -> str:
    try:
        img = load_coa_logo(path)
        if img is not None:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            data = buf.getvalue()
        else:
            with open(path, "rb") as f:
                data = f.read()
        b64 = base64.b64encode(data).decode("ascii")
        return f"<img src='data:image/png;base64,{b64}' style='width:{width_px}px'/>"
    except Exception as e:
        logger.info(f"Logo tag failed: {e}")
        return ""

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
        color-scheme: dark;
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
        background-color: var(--background) !important;
        color: var(--text-primary) !important;
    }}
    
    .stImage {{
        display: flex;
        justify-content: center;
    }}
    .stImage img {{
        display: block;
        margin: 0 auto;
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

class UpdateNote(Base):
    __tablename__ = 'updates'
    id = Column(Integer, primary_key=True)
    year = Column(Integer, nullable=False)
    month = Column(Integer, nullable=False)
    strategy_id = Column(Integer, nullable=True)
    note = Column(Text, nullable=False)
    created_by = Column(String, nullable=False)
    created_at = Column(Date, default=datetime.date.today)

if DB_URL.startswith('sqlite'):
    engine = create_engine(DB_URL, connect_args={"check_same_thread": False})
else:
    engine = create_engine(DB_URL)

try:
    Base.metadata.create_all(engine)
except Exception:
    pass
session_factory = sessionmaker(bind=engine)
Session = scoped_session(session_factory)

pwd_context = CryptContext(schemes=["bcrypt", "pbkdf2_sha256"], deprecated="auto")

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
@st.cache_data(show_spinner=False, ttl=3600)
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

def get_active_strategies_df() -> pd.DataFrame:
    with get_db_session() as db:
        return pd.read_sql(
            db.query(Strategy).filter(Strategy.is_active == True).statement,
            db.bind
        )

def get_protocol_options(strategies_df: pd.DataFrame, include_no: bool = False, include_all: bool = False) -> list:
    options = []
    if include_all:
        options.append('All Protocols')
    if include_no:
        options.append('No Protocol')
    if strategies_df is not None and not strategies_df.empty:
        options.extend(sorted(strategies_df['name'].tolist()))
    return options

def resolve_strategy_id_by_name(strategies_df: pd.DataFrame, name: str):
    if strategies_df is not None and not strategies_df.empty and name:
        match = strategies_df[strategies_df['name'] == name]
        if not match.empty:
            return int(match['id'].iloc[0])
    return None

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
            if strategy_id is None and pd.notna(row.get('strategy_id')):
                continue
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

def compute_aggregated_portfolio_history(all_events_df: pd.DataFrame, strategy_data: dict):
    if not strategy_data:
        return None, pd.DataFrame(), None
    histories = {name: d['history'][['date', 'total']].copy() for name, d in strategy_data.items() if not d['history'].empty}
    if not histories:
        return None, pd.DataFrame(), None
    for k in histories:
        histories[k]['date'] = pd.to_datetime(histories[k]['date'])
    all_dates = sorted(pd.to_datetime(pd.concat([h['date'] for h in histories.values()]).unique()))
    agg_df = pd.DataFrame({'date': all_dates})
    proto_cols = []
    for name, h in histories.items():
        col_name = name
        proto_cols.append(col_name)
        tmp = h.rename(columns={'total': col_name})
        agg_df = agg_df.merge(tmp, on='date', how='left')
    for c in proto_cols:
        agg_df[c] = agg_df[c].ffill().fillna(0.0)
    agg_df['Total'] = agg_df[proto_cols].sum(axis=1)
    dep_wd = all_events_df[all_events_df['type'].isin(['deposit', 'withdrawal'])].copy()
    if not dep_wd.empty:
        dep_wd['date'] = pd.to_datetime(dep_wd['date'])
    union_dates = sorted(pd.to_datetime(pd.unique(pd.concat([agg_df['date'], dep_wd['date'] if not dep_wd.empty else pd.Series([])]) )))
    balances = {}
    rows = []
    for dt in union_dates:
        if not dep_wd.empty:
            todays = dep_wd[dep_wd['date'] == dt]
            for _, r in todays.iterrows():
                inv = r.get('investor')
                usd_amount = float(r.get('usd_amount', 0.0) or 0.0)
                if r.get('type') == 'deposit' and inv and usd_amount > 0:
                    balances[inv] = balances.get(inv, 0.0) + usd_amount
                elif r.get('type') == 'withdrawal' and inv and usd_amount > 0:
                    balances[inv] = balances.get(inv, 0.0) - usd_amount
        if dt in set(agg_df['date']):
            target_total = float(agg_df.loc[agg_df['date'] == dt, 'Total'].iloc[0])
            current_total = sum(balances.values())
            if current_total > 0 and target_total > 0:
                factor = target_total / current_total
                for inv in list(balances.keys()):
                    balances[inv] = balances[inv] * factor
        snap = {'date': dt, 'total': sum(balances.values())}
        for inv, val in balances.items():
            snap[inv] = val
        rows.append(snap)
    history_df = pd.DataFrame(rows) if rows else pd.DataFrame()
    return balances, history_df, agg_df

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

# ---------- Investor Annual Performance ----------
def calculate_annual_performance(investor_name: str, all_events_df: pd.DataFrame, portfolio_history_df: pd.DataFrame = None) -> pd.DataFrame:
    if not investor_name or all_events_df is None or all_events_df.empty:
        return pd.DataFrame()
    # Check all events for the investor to determine start year, not just deposits
    inv_events = all_events_df[all_events_df['investor'] == investor_name]
    if inv_events.empty:
        return pd.DataFrame()
    first_year = pd.to_datetime(inv_events['date']).dt.year.min()
    current_year = datetime.date.today().year
    records = []
    for year in range(int(first_year), int(current_year) + 1):
        start_date = datetime.date(year, 1, 1)
        end_date = datetime.date(year, 12, 31)
        if end_date > datetime.date.today():
            end_date = datetime.date.today()
        # For annual performance, we can try to use portfolio history for speed, but fallback is safer
        # Logic remains similar but we ensure first_year is correct
        if portfolio_history_df is not None and not portfolio_history_df.empty and investor_name in portfolio_history_df.columns:
            ph = portfolio_history_df.copy()
            ph['date'] = pd.to_datetime(ph['date']).dt.date
            before_start = ph[ph['date'] < start_date]
            val = before_start[investor_name].iloc[-1] if not before_start.empty else 0.0
            start_balance = float(val) if pd.notna(val) else 0.0
        else:
            events_upto_start = all_events_df[all_events_df['date'] < start_date]
            start_balances, _ = replay_events(events_upto_start)
            start_balance = float(start_balances.get(investor_name, 0.0))
        year_events = all_events_df[(all_events_df['date'] >= start_date) & (all_events_df['date'] <= end_date)]
        deposits = year_events[(year_events['investor'] == investor_name) & (year_events['type'] == 'deposit')]['usd_amount'].sum()
        withdrawals = year_events[(year_events['investor'] == investor_name) & (year_events['type'] == 'withdrawal')]['usd_amount'].sum()
        if portfolio_history_df is not None and not portfolio_history_df.empty and investor_name in portfolio_history_df.columns:
            ph = portfolio_history_df.copy()
            ph['date'] = pd.to_datetime(ph['date']).dt.date
            upto_end = ph[ph['date'] <= end_date]
            val = upto_end[investor_name].iloc[-1] if not upto_end.empty else start_balance
            end_balance = float(val) if pd.notna(val) else start_balance
        else:
            events_upto_end = all_events_df[all_events_df['date'] <= end_date]
            end_balances, _ = replay_events(events_upto_end)
            end_balance = float(end_balances.get(investor_name, 0.0))
        start_of_year = float(start_balance) + float(deposits or 0.0)
        net_gain = end_balance - start_balance - float(deposits or 0.0) + float(withdrawals or 0.0)
        roi_pct = (net_gain / float(start_of_year) * 100) if float(start_of_year) > 0 else None
        records.append({
            'Year': year,
            'Start_Value': start_balance,
            'Start_Year_Balance': float(start_balance),
            'Start_of_Year': float(start_of_year),
            'End_Value': end_balance,
            'Deposits': float(deposits or 0.0),
            'Withdrawals': float(withdrawals or 0.0),
            'Net_Gain': float(net_gain),
            'ROI %': float(roi_pct) if roi_pct is not None else None
        })
    return pd.DataFrame(records)

def calculate_monthly_performance(investor_name: str, year: int, all_events_df: pd.DataFrame, portfolio_history_df: pd.DataFrame = None) -> pd.DataFrame:
    if not investor_name or all_events_df is None or all_events_df.empty:
        return pd.DataFrame()
    year = int(year)
    today = datetime.date.today()
    if datetime.date(year, 1, 1) > today:
        return pd.DataFrame()
    events_df = all_events_df.copy()
    events_df['date'] = pd.to_datetime(events_df['date']).dt.date
    month_names = {1:'Gennaio',2:'Febbraio',3:'Marzo',4:'Aprile',5:'Maggio',6:'Giugno',7:'Luglio',8:'Agosto',9:'Settembre',10:'Ottobre',11:'Novembre',12:'Dicembre'}
    
    # Use portfolio_history if available to capture market value changes (unrealized gains)
    ph = None
    ph_last_date = None
    if portfolio_history_df is not None and not portfolio_history_df.empty and investor_name in portfolio_history_df.columns:
        ph = portfolio_history_df.copy()
        ph['date'] = pd.to_datetime(ph['date']).dt.date
        ph_last_date = ph['date'].max()
    
    records = []
    for month in range(1, 13):
        start_date = datetime.date(year, month, 1)
        if start_date > today:
            break
        end_date = datetime.date(year, month, calendar.monthrange(year, month)[1])
        if end_date > today:
            end_date = today
        if ph is not None:
            before_start = ph[ph['date'] < start_date]
            val = before_start[investor_name].iloc[-1] if not before_start.empty else 0.0
            start_balance = float(val) if pd.notna(val) else 0.0
        else:
            events_upto_start = events_df[events_df['date'] < start_date]
            start_balances, _ = replay_events(events_upto_start)
            start_balance = float(start_balances.get(investor_name, 0.0))
            
        # Determine if we can use PH for end_balance or need to replay (e.g. if PH is stale)
        use_ph_end = False
        if ph is not None and ph_last_date is not None:
            if ph_last_date >= end_date:
                use_ph_end = True
                
        if use_ph_end:
            upto_end = ph[ph['date'] <= end_date]
            val = upto_end[investor_name].iloc[-1] if not upto_end.empty else start_balance
            end_balance = float(val) if pd.notna(val) else start_balance
        else:
            events_upto_end = events_df[events_df['date'] <= end_date]
            end_balances, _ = replay_events(events_upto_end)
            end_balance = float(end_balances.get(investor_name, 0.0))
        month_events = events_df[(events_df['date'] >= start_date) & (events_df['date'] <= end_date) & (events_df['investor'] == investor_name)]
        deposits = month_events[month_events['type'] == 'deposit']['usd_amount'].sum()
        withdrawals = month_events[month_events['type'] == 'withdrawal']['usd_amount'].sum()
        net_gain = end_balance - start_balance - float(deposits or 0.0) + float(withdrawals or 0.0)
        start_of_month = float(start_balance) + float(deposits or 0.0)
        roi_pct = (net_gain / float(start_of_month) * 100) if float(start_of_month) > 0 else None
        records.append({
            'Mese_Num': month,
            'Mese': month_names.get(month, str(month)),
            'Valore Inizio': float(start_balance),
            'Valore Fine': float(end_balance),
            'Depositi': float(deposits or 0.0),
            'Prelievi': float(withdrawals or 0.0),
            'Guadagno Netto': float(net_gain),
            'ROI %': float(roi_pct) if roi_pct is not None else None
        })
    return pd.DataFrame(records)

def display_annual_chart(annual_df: pd.DataFrame, title: str):
    if annual_df is None or annual_df.empty:
        return
    years = [str(int(y)) for y in annual_df['Year'].tolist()]
    gains = [float(g) if pd.notna(g) else 0.0 for g in annual_df['Net_Gain'].tolist()]
    rois = [float(r) if pd.notna(r) else 0.0 for r in annual_df['ROI %'].tolist()] if 'ROI %' in annual_df.columns else [0.0 for _ in gains]
    colors = [COA_COLORS['primary_blue'] if (g or 0) >= 0 else COA_COLORS['primary_purple'] for g in gains]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=years,
        y=gains,
        marker_color=colors,
        text=[f"${g:,.0f} | {r:+.1f}%" for g, r in zip(gains, rois)],
        textposition='outside',
        textfont=dict(size=16),
        texttemplate='<b>%{text}</b>',
        cliponaxis=False
    ))
    fig.update_layout(
        title=title,
        xaxis_title='Anno',
        yaxis_title='Guadagno Netto (USD)',
        yaxis_tickformat='$,.0f',
        xaxis_type='category',
        xaxis_categoryorder='category ascending',
        plot_bgcolor='#1a1a1a',
        paper_bgcolor='#1a1a1a',
        font=dict(color='#e2e8f0'),
        height=420
    )
    fig.update_xaxes(gridcolor='rgba(226,232,240,0.15)')
    fig.update_yaxes(gridcolor='rgba(226,232,240,0.15)', zerolinecolor='rgba(226,232,240,0.25)')
    fig.update_traces(textfont_color='#e2e8f0')
    st.plotly_chart(fig, use_container_width=True)

def display_monthly_chart(monthly_df: pd.DataFrame, title: str):
    if monthly_df is None or monthly_df.empty:
        return
    monthly_df = monthly_df.sort_values('Mese_Num')
    months = monthly_df['Mese'].tolist()
    gains = [float(g) if pd.notna(g) else 0.0 for g in monthly_df['Guadagno Netto'].tolist()]
    rois = [float(r) if pd.notna(r) else 0.0 for r in monthly_df['ROI %'].tolist()] if 'ROI %' in monthly_df.columns else [0.0 for _ in gains]
    colors = [COA_COLORS['primary_blue'] if (g or 0) >= 0 else COA_COLORS['primary_purple'] for g in gains]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=months,
        y=gains,
        marker_color=colors,
        text=[f"${g:,.0f} | {r:+.1f}%" for g, r in zip(gains, rois)],
        textposition='outside',
        textfont=dict(size=14),
        texttemplate='<b>%{text}</b>',
        cliponaxis=False
    ))
    fig.update_layout(
        title=title,
        xaxis_title='Mese',
        yaxis_title='Guadagno Netto (USD)',
        yaxis_tickformat='$,.0f',
        xaxis_type='category',
        plot_bgcolor='#1a1a1a',
        paper_bgcolor='#1a1a1a',
        font=dict(color='#e2e8f0'),
        height=420
    )
    fig.update_xaxes(gridcolor='rgba(226,232,240,0.15)')
    fig.update_yaxes(gridcolor='rgba(226,232,240,0.15)', zerolinecolor='rgba(226,232,240,0.25)')
    fig.update_traces(textfont_color='#e2e8f0')
    st.plotly_chart(fig, use_container_width=True)

def display_multi_investor_annual_chart(investors: list, all_events_df: pd.DataFrame, portfolio_history_df: pd.DataFrame = None):
    if not investors:
        return
    per_inv = {}
    all_years = set()
    for inv in investors:
        df = calculate_annual_performance(inv, all_events_df, portfolio_history_df)
        if df is not None and not df.empty:
            if 'Start_of_Year' not in df.columns and 'Start_Year_Balance' in df.columns:
                df['Start_of_Year'] = pd.to_numeric(df['Start_Year_Balance'], errors='coerce').fillna(0.0) + pd.to_numeric(df['Deposits'], errors='coerce').fillna(0.0)
            # Recompute Net_Gain defensively to avoid cache inconsistencies
            df['__NG_FIX__'] = (
                pd.to_numeric(df['End_Value'], errors='coerce').fillna(0.0)
                - pd.to_numeric(df['Start_Year_Balance'], errors='coerce').fillna(0.0)
                - pd.to_numeric(df['Deposits'], errors='coerce').fillna(0.0)
                + pd.to_numeric(df['Withdrawals'], errors='coerce').fillna(0.0)
            )
            per_inv[inv] = df[['Year', '__NG_FIX__']].rename(columns={'__NG_FIX__': 'Net_Gain'}).copy()
            for y in df['Year'].tolist():
                all_years.add(int(y))
    if not per_inv:
        return
    years_sorted = sorted(list(all_years))
    years_sorted_str = [str(int(y)) for y in years_sorted]
    fig = go.Figure()
    palette = px.colors.qualitative.Plotly
    inv_colors = {inv: palette[i % len(palette)] for i, inv in enumerate(sorted(per_inv.keys()))}
    for inv, df in per_inv.items():
        gains_map = {int(r['Year']): (float(r['Net_Gain']) if pd.notna(r['Net_Gain']) else 0.0) for _, r in df.iterrows()}
        gains = [gains_map.get(y, 0.0) for y in years_sorted]
        fig.add_trace(go.Bar(x=years_sorted_str, y=gains, name=inv, marker_color=inv_colors.get(inv, COA_COLORS['primary_blue']), text=[f"${g:,.0f}" for g in gains], textposition='outside', cliponaxis=False))
    fig.update_layout(
        barmode='group',
        title='Guadagni Annuali - Tutti gli investitori',
        xaxis_title='Anno',
        yaxis_title='Guadagno Netto (USD)',
        yaxis_tickformat='$,.0f',
        xaxis_type='category',
        plot_bgcolor='#1a1a1a',
        paper_bgcolor='#1a1a1a',
        font=dict(color='#e2e8f0'),
        height=420
    )
    fig.update_xaxes(gridcolor='rgba(226,232,240,0.15)')
    fig.update_yaxes(gridcolor='rgba(226,232,240,0.15)', zerolinecolor='rgba(226,232,240,0.25)')
    fig.update_traces(textfont_color='#e2e8f0')
    st.plotly_chart(fig, use_container_width=True)

# ---------- Main App ----------
st.set_page_config(page_title='COA Equity Tracker', page_icon='📊', layout='wide')

if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'

apply_coa_styling()

# Initialize session state
if 'jwt' not in st.session_state:
    st.session_state.jwt, st.session_state.username, st.session_state.role = None, None, None

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
        _img_tag = get_logo_img_tag(width_px=140)
        st.markdown(f"""
        <div style="display:flex; flex-direction:column; align-items:center;">
            {_img_tag}
            <h2 style="margin-top: 0.6rem; margin-bottom: 1.6rem;">COA-Portfolio</h2>
        </div>
        """, unsafe_allow_html=True)
        
        with st.form("login_form"):
            login_user = st.text_input('Nome Utente', placeholder='Inserisci il tuo nome utente')
            login_pw = st.text_input('Password', type='password', placeholder='Inserisci la tua password')
            
            col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
            with col_btn2:
                if st.form_submit_button('Accedi', use_container_width=True):
                    with get_db_session() as db:
                        lu = (login_user or "").strip()
                        lp = (login_pw or "").strip()
                        if not lu or not lp:
                            st.error('❌ Inserisci username e password')
                        else:
                            user = db.query(User).filter(func.lower(User.username) == lu.lower()).first()
                            if not user:
                                st.error('❌ Utente non trovato')
                            elif verify_password(lp, user.password_hash):
                                st.session_state.jwt, st.session_state.username, st.session_state.role = create_jwt(user.username, user.role), user.username, user.role
                                st.rerun()
                            else:
                                st.error('❌ Password errata')
        
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
    st.error('⏰ Sessione non valida o scaduta. Effettua nuovamente il login.')
    st.session_state.clear()
    if st.button('🔙 Torna al Login'): st.rerun()
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
            render_logo_centered(width_px=160)
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
                <h1>COA-Portfolio</h1>
                <p>Gestione e analisi del portafoglio COA</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div style="background: transparent; padding: 1rem; border-radius: 12px; box-shadow: none; text-align: center;">
        <div style="color: {COA_COLORS['primary_purple']}; font-weight: 600; margin-bottom: 0.5rem;">
            👤 {current_user}
        </div>
        <div style="color: {COA_COLORS['text_secondary']}; font-size: 0.9rem; margin-bottom: 1rem;">
            {current_role.title()}
        </div>
        {f'<div style="color: {COA_COLORS["success"]}; font-size: 0.8rem;">✓ Active</div>' if payload else ''}
    </div>
    """, unsafe_allow_html=True)
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
    strategies_df = pd.read_sql(db.query(Strategy).filter(Strategy.is_active == True).statement, db.bind)

    updates_df = pd.read_sql(db.query(UpdateNote).statement, db.bind)

 

# ---------- Main Dashboard ----------
if all_events_df.empty:
    st.info('📊 Nessun evento nel sistema. Inizia aggiungendo il tuo primo deposito!')
    
    if current_role == 'admin':
        with st.form("first_deposit_form"):
            st.subheader("➕ Aggiungi Primo Evento")
            col1, col2 = st.columns(2)
            
            with col1:
                d_date = st.date_input('Data', datetime.date.today())
                investor = st.text_input('Nome Investitore', placeholder='Inserisci nome investitore')
                
            with col2:
                protocol_options = ['Nessun Protocollo'] + (sorted(strategies_df['name'].tolist()) if not strategies_df.empty else [])
                selected_protocol = st.selectbox('Protocollo', protocol_options)
                
            eur_amount = st.number_input('Importo (EUR)', min_value=0.01, step=100.0, value=1000.0)
            
            if st.form_submit_button('💰 Aggiungi Primo Deposito', use_container_width=True):
                rate = get_historical_eurusd(d_date)
                usd_amount = eur_amount * rate
                
                with get_db_session() as db:
                    strategy_id = resolve_strategy_id_by_name(strategies_df, selected_protocol) if selected_protocol != 'Nessun Protocollo' else None
                    
                    db.add(Event(
                        date=d_date, 
                        type='deposit', 
                        strategy_id=strategy_id,
                        investor=investor.strip(), 
                        eur_amount=eur_amount, 
                        usd_amount=usd_amount, 
                        eurusd_rate=rate
                    ))
                
                st.success('🎉 Primo deposito aggiunto con successo!')
                time.sleep(2)
                st.rerun()
    else:
        st.stop()

else:
    # Filter events based on user role and strategy selection
    
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

    total_balances, total_history = replay_events(all_events_df)
    total_portfolio_value = sum(total_balances.values())
    
    # Calculate strategy-specific data
    strategy_data = {}
    if strategies_df is not None and not strategies_df.empty:
        for _, strat in strategies_df.iterrows():
            strat_balances, strat_history = replay_events(all_events_df, int(strat['id']))
            strategy_data[strat['name']] = {
                'balances': strat_balances,
                'history': strat_history,
                'total_value': sum(strat_balances.values())
            }
    strategies_with_funds = {name: data for name, data in strategy_data.items() if data.get('total_value', 0.0) > 0}

    aggregated_total = None
    agg_balances, agg_history_df, agg_df_tmp = compute_aggregated_portfolio_history(all_events_df, strategy_data)
    if agg_df_tmp is not None and not agg_df_tmp.empty:
        aggregated_total = float(agg_df_tmp['Total'].iloc[-1])

    if aggregated_total is not None:
        total_portfolio_value = aggregated_total
        if agg_history_df is not None and not agg_history_df.empty:
            total_history = agg_history_df.copy()
        if agg_balances is not None:
            total_balances = agg_balances
    
    # Calculate overall ROI
    overall_roi = 0
    if total_portfolio_value > 0:
        total_usd_invested_overall = all_events_df[all_events_df['type'] == 'deposit']['usd_amount'].sum()
        if total_usd_invested_overall > 0:
            overall_roi = ((total_portfolio_value - total_usd_invested_overall) / total_usd_invested_overall * 100)

    # ---------- Key Metrics ----------
    st.markdown("### 📊 Portafoglio Overview")
    
    # Main metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Valore Totale Portafoglio</div>
            <div class="metric-value">${total_portfolio_value:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">ROI Complessivo</div>
            <div class="metric-value" style="color: {COA_COLORS['primary_blue'] if overall_roi >= 0 else COA_COLORS['primary_purple']}">
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
                <div class="metric-label">La Tua Quota</div>
                <div class="metric-value">{user_share:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            visible_investors = sorted({inv for inv in events_df['investor'].dropna().unique()})
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Investitori Attivi</div>
                <div class="metric-value">{len(visible_investors)}</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col4:
        if strategies_df is not None and not strategies_df.empty:
            active_strategies = len(strategies_with_funds)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Strategie Attive</div>
                <div class="metric-value">{active_strategies}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Strategie</div>
                <div class="metric-value">Nessuna</div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # ---------- Navigation Tabs ----------
    tab_list = ["📈 Dashboard", "👥 Dettagli Investitore", "📝 Aggiornamenti"]
    if current_role == 'admin':
        tab_list.extend(["⚙️ Gestione Eventi"])
    
    tabs = st.tabs(tab_list)

    # Dashboard Tab
    with tabs[0]:
        st.markdown("### 📈 Performance Portfolio e Protocolli")
        
        col1, col2 = st.columns([3, 1])
        proto_color_seed = {'TS Futures': COA_COLORS['primary_blue'], 'Seasonal Stock': COA_COLORS['light_purple']}
        
        # Define a pool of colors, excluding those already used in the seed to prevent duplicates
        base_pool = [
            COA_COLORS['primary_purple'], 
            COA_COLORS['primary_blue'], 
            COA_COLORS['light_purple'], 
            COA_COLORS['light_blue'],
            '#00C853', '#FFAB00', '#D500F9', '#2962FF', '#FF3D00', '#00E5FF', '#76FF03',
            '#C2185B', '#7B1FA2', '#512DA8', '#303F9F', '#1976D2', '#0288D1', '#0097A7',
            '#00796B', '#388E3C', '#689F38', '#AFB42B', '#FBC02D', '#FFA000', '#F57C00',
            '#E64A19', '#5D4037', '#616161', '#455A64'
        ]
        used_seed_colors = set(proto_color_seed.values())
        available_colors = [c for c in base_pool if c not in used_seed_colors]
        
        proto_color_map = {}
        if agg_df_tmp is not None and not agg_df_tmp.empty:
            proto_cols_all = [c for c in agg_df_tmp.columns if c not in ['date', 'Total']]
            
            # First pass: assign seed colors
            for c in proto_cols_all:
                if c in proto_color_seed:
                    proto_color_map[c] = proto_color_seed[c]
            
            # Second pass: assign remaining colors
            for c in proto_cols_all:
                if c not in proto_color_map:
                    if available_colors:
                        proto_color_map[c] = available_colors.pop(0)
                    else:
                        # Fallback if we run out of colors (unlikely with expanded pool)
                        proto_color_map[c] = COA_COLORS['text_secondary']

        with col1:
            # Main portfolio chart
            if agg_df_tmp is not None and not agg_df_tmp.empty:
                agg_df = agg_df_tmp.copy()
                proto_cols = [c for c in agg_df.columns if c not in ['date','Total']]
                fig_portfolio = go.Figure()
                fig_portfolio.add_trace(go.Scatter(
                    x=agg_df['date'],
                    y=agg_df['Total'],
                    name='Portfolio',
                    mode='lines+markers',
                    line=dict(color='#ffffff', width=3),
                    marker=dict(size=6, color='#ffffff')
                ))
                for i, c in enumerate(proto_cols):
                    col = proto_color_map.get(c, COA_COLORS['primary_purple'])
                    fig_portfolio.add_trace(go.Scatter(
                        x=agg_df['date'],
                        y=agg_df[c],
                        name=c,
                        mode='lines+markers',
                        line=dict(width=2, color=col),
                        marker=dict(size=5, color=col)
                    ))
                fig_portfolio.update_layout(
                    title='Performance Protocolli',
                    plot_bgcolor='#1a1a1a',
                    paper_bgcolor='#1a1a1a',
                    font=dict(color='#e2e8f0'),
                    title_font_size=16,
                    height=560,
                    hovermode='x unified',
                    xaxis=dict(showgrid=True, showline=True, showticklabels=True, zeroline=False),
                    yaxis=dict(showgrid=True, showline=True, showticklabels=True, zeroline=False, tickformat='$,.0f')
                )
                fig_portfolio.update_xaxes(gridcolor='rgba(226,232,240,0.15)')
                fig_portfolio.update_yaxes(gridcolor='rgba(226,232,240,0.15)', zerolinecolor='rgba(226,232,240,0.25)')
                st.plotly_chart(fig_portfolio, use_container_width=True)
            elif not total_history.empty:
                fig_portfolio = px.line(
                    total_history,
                    x='date',
                    y='total',
                    title='Performance Protocolli',
                    labels={'date': 'Date', 'total': 'Portfolio Value (USD)'},
                    markers=True,
                    render_mode='svg'
                )
                fig_portfolio.update_traces(
                    line_color=COA_COLORS['primary_purple'],
                    line_width=3,
                    marker_size=6,
                    marker_color=COA_COLORS['primary_blue']
                )
                fig_portfolio.for_each_trace(lambda t: t.update(name='Portfolio'))
                fig_portfolio.update_layout(
                    plot_bgcolor='#1a1a1a',
                    paper_bgcolor='#1a1a1a',
                    font=dict(color='#e2e8f0'),
                    title_font_size=16,
                    title_font_color=COA_COLORS['primary_purple'],
                    height=560,
                    hovermode='x unified',
                    xaxis=dict(showgrid=True, showline=True, showticklabels=True, zeroline=False),
                    yaxis=dict(showgrid=True, showline=True, showticklabels=True, zeroline=False, tickformat='$,.0f')
                )
                fig_portfolio.update_xaxes(gridcolor='rgba(226,232,240,0.15)')
                fig_portfolio.update_yaxes(gridcolor='rgba(226,232,240,0.15)', zerolinecolor='rgba(226,232,240,0.25)')
                st.plotly_chart(fig_portfolio, use_container_width=True)
            else:
                st.info("Nessun dato storico ancora disponibile")
        
        with col2:
            # Strategy performance comparison
            if agg_df_tmp is not None and not agg_df_tmp.empty:
                agg_df_pie = agg_df_tmp.copy()
                proto_cols_pie = [c for c in agg_df_pie.columns if c not in ['date','Total']]
                alloc_names = proto_cols_pie
                alloc_values = agg_df_pie[proto_cols_pie].iloc[-1].tolist() if not agg_df_pie.empty else []
                fig_strategies = go.Figure(data=[go.Pie(
                    labels=alloc_names,
                    values=alloc_values,
                    marker=dict(colors=[proto_color_map.get(n, COA_COLORS['primary_purple']) for n in alloc_names])
                )])
                fig_strategies.update_layout(
                    title='Allocazione Protocolli',
                    height=460,
                    plot_bgcolor='#1a1a1a',
                    paper_bgcolor='#1a1a1a',
                    font=dict(color='#e2e8f0')
                )
                st.plotly_chart(fig_strategies, use_container_width=True)
        
        # Add shared charts like in CHIODI_old.py
        if current_role == 'admin':
            st.subheader("📈 Valori delle Quote Individuali (USD) nel Tempo")
            if not total_history.empty and len(total_history.columns) > 2:
                investor_cols = [c for c in total_history.columns if c not in ['date', 'total']]
                if investor_cols:
                    value_df_melted = total_history.melt(id_vars='date', value_vars=investor_cols, var_name='Investitore', value_name='Valore (USD)')
                    fig_investor_value = px.line(value_df_melted, x='date', y='Valore (USD)', color='Investitore', markers=True, 
                                               labels={'Valore (USD)': 'Valore (USD)', 'date': 'Data'})
                    fig_investor_value.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig_investor_value, use_container_width=True)
        else: 
            st.subheader("📈 Andamento del Tuo Valore (USD)")
            if user_investor_name and user_investor_name in total_history.columns:
                user_history_df = total_history[['date', user_investor_name]].copy()
                user_history_df.rename(columns={user_investor_name: 'Il Tuo Valore (USD)'}, inplace=True)
                fig_user_value = px.line(user_history_df, x='date', y='Il Tuo Valore (USD)', markers=True)
                st.plotly_chart(fig_user_value, use_container_width=True)
        
        st.subheader("📊 Evoluzione Quote Investitori (%)")
        history_to_show = total_history.copy()
        if current_role != 'admin':
            if user_investor_name and user_investor_name in history_to_show.columns:
                history_to_show = history_to_show[['date', 'total', user_investor_name]].copy()
        if not history_to_show.empty and len(history_to_show.columns) > 2:
            inv_cols = [c for c in history_to_show.columns if c not in ['date', 'total']]
            if inv_cols:
                pct_df = history_to_show[inv_cols].div(history_to_show['total'], axis=0).fillna(0)
                pct_df['date'] = history_to_show['date']
                pct_df_melted = pct_df.melt(id_vars='date', var_name='Investitore', value_name='Quota')
                fig_shares = px.area(pct_df_melted, x='date', y='Quota', color='Investitore', markers=True)
                fig_shares.update_yaxes(tickformat='.0%')
                fig_shares.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_shares, use_container_width=True)

    # Investor Details Tab
    with tabs[1]:
        st.markdown("### 👥 Performance Investitori")
        try:
            st.cache_data.clear()
        except Exception:
            pass
        
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
                if pd.isna(val):
                    return 'background-color: rgba(160,174,192,0.15); color: var(--text-primary);'
                intensity = min(abs(float(val)) / 50.0, 1.0)
                alpha = 0.18 + 0.32 * intensity
                if float(val) > 0:
                    return f'background-color: rgba(30,140,200,{alpha}); color: white; font-weight: 600;'
                elif float(val) < 0:
                    return f'background-color: rgba(122,46,143,{alpha}); color: white; font-weight: 600;'
                else:
                    return 'background-color: rgba(160,174,192,0.15); color: var(--text-primary);'
            
            display_df = df_inv.rename(columns={
                'Investor': 'Investitore',
                'EUR Invested': 'Investito EUR',
                'USD Invested': 'Investito USD',
                'Total Withdrawn (USD)': 'Prelevato Totale (USD)',
                'Current Value (USD)': 'Valore Attuale (USD)',
                'Share %': 'Quota %',
                'ROI %': 'ROI %'
            })
            styled_df = display_df.style.format({
                'Investito EUR': '€{:,.2f}',
                'Investito USD': '${:,.2f}',
                'Prelevato Totale (USD)': '${:,.2f}',
                'Valore Attuale (USD)': '${:,.2f}',
                'Quota %': '{:.2f}%',
                'ROI %': '{:+.2f}%'
            }).apply(lambda x: x.map(roi_color), subset=['ROI %'])
            
            st.dataframe(styled_df, use_container_width=True, hide_index=True)

        st.divider()
        if current_role == 'admin':
            st.subheader('📈 Storico Performance Annuale')
            options = ['Tutti gli investitori'] + investors_to_show
            selected_investor = st.selectbox('Seleziona investitore per vista annuale', options)
            if selected_investor == 'Tutti gli investitori':
                display_multi_investor_annual_chart(investors_to_show, all_events_df, total_history)
                combined_rows = []
                for inv in investors_to_show:
                    df = calculate_annual_performance(inv, all_events_df, total_history)
                    if df is not None and not df.empty:
                        sub = df[['Year','Deposits','Withdrawals','End_Value']].copy()
                        sub.rename(columns={'End_Value': 'Year_End_Value'}, inplace=True)
                        sub['Investor'] = inv
                        combined_rows.append(sub)
                if combined_rows:
                    combined = pd.concat(combined_rows).sort_values(['Investor','Year'])
                    combined = combined.rename(columns={
                        'Investor': 'Investitore',
                        'Year': 'Anno',
                        'Deposits': 'Depositi',
                        'Withdrawals': 'Prelievi',
                        'Year_End_Value': 'Valore Fine Anno'
                    })
                    st.subheader('💰 Flussi Annuali (Tutti gli investitori)')
                    for c in ['Depositi','Prelievi','Valore Fine Anno']:
                        combined[c] = pd.to_numeric(combined[c], errors='coerce').fillna(0.0)
                    st.dataframe(
                        combined.style.format({'Depositi': '${:,.0f}', 'Prelievi': '${:,.0f}', 'Valore Fine Anno': '${:,.0f}'}),
                        use_container_width=True,
                        hide_index=True
                    )
                st.subheader('📆 Guadagni Mensili')
                st.info("Seleziona un investitore specifico per vedere i guadagni mensili.")
            else:
                annual_df = calculate_annual_performance(selected_investor, all_events_df, total_history)
                if annual_df is not None and not annual_df.empty:
                    inv_norm = selected_investor.strip().lower()
                    th_cols_norm = {str(c).strip().lower(): c for c in total_history.columns if c not in ['date','total']} if (total_history is not None and not total_history.empty) else {}
                    th_col = th_cols_norm.get(inv_norm)
                    if total_history is not None and not total_history.empty and th_col is not None:
                        ph_fix = total_history.copy()
                        ph_fix['date'] = pd.to_datetime(ph_fix['date']).dt.date
                        for i, r in annual_df.iterrows():
                            y = int(r['Year'])
                            end_dt = datetime.date(y, 12, 31)
                            upto_end = ph_fix[ph_fix['date'] <= end_dt]
                            if not upto_end.empty:
                                end_bal_fix = float(upto_end[th_col].iloc[-1])
                            else:
                                events_upto_end = all_events_df[pd.to_datetime(all_events_df['date']).dt.date <= end_dt]
                                end_balances, _ = replay_events(events_upto_end)
                                kb_map = {str(k).strip().lower(): k for k in end_balances.keys()}
                                end_bal_fix = float(end_balances.get(kb_map.get(inv_norm, selected_investor), 0.0))
                            ev = all_events_df.copy()
                            ev['__inv_norm__'] = ev['investor'].astype(str).str.strip().str.lower()
                            year_mask = (pd.to_datetime(ev['date']).dt.year == y) & (ev['__inv_norm__'] == inv_norm)
                            deposits_fix = float(all_events_df[year_mask & (all_events_df['type'] == 'deposit')]['usd_amount'].sum() or 0.0)
                            withdrawals_fix = float(all_events_df[year_mask & (all_events_df['type'] == 'withdrawal')]['usd_amount'].sum() or 0.0)
                            start_bal_fix = float(r['Start_Year_Balance'] or 0.0)
                            start_of_year_fix = start_bal_fix + deposits_fix
                            net_gain_fix = end_bal_fix - start_bal_fix - deposits_fix + withdrawals_fix
                            roi_fix = float(net_gain_fix / start_of_year_fix * 100) if start_of_year_fix > 0 else None
                            annual_df.at[i, 'End_Value'] = float(end_bal_fix)
                            annual_df.at[i, 'Start_of_Year'] = float(start_of_year_fix)
                            annual_df.at[i, 'Deposits'] = float(deposits_fix)
                            annual_df.at[i, 'Withdrawals'] = float(withdrawals_fix)
                            annual_df.at[i, 'Net_Gain'] = float(net_gain_fix)
                            annual_df.at[i, 'ROI %'] = roi_fix
                    annual_df['Start_of_Year'] = pd.to_numeric(annual_df.get('Start_Year_Balance', 0.0), errors='coerce').fillna(0.0) + pd.to_numeric(annual_df.get('Deposits', 0.0), errors='coerce').fillna(0.0)
                    annual_df['Net_Gain'] = (
                        pd.to_numeric(annual_df.get('End_Value', 0.0), errors='coerce').fillna(0.0)
                        - pd.to_numeric(annual_df.get('Start_Year_Balance', annual_df.get('Start_Value', 0.0)), errors='coerce').fillna(0.0)
                        - pd.to_numeric(annual_df.get('Deposits', 0.0), errors='coerce').fillna(0.0)
                        + pd.to_numeric(annual_df.get('Withdrawals', 0.0), errors='coerce').fillna(0.0)
                    )
                    annual_df['ROI %'] = annual_df.apply(lambda r: (float(r['Net_Gain']) / float(r['Start_of_Year']) * 100) if float(r['Start_of_Year']) > 0 else None, axis=1)
                    display_annual_chart(annual_df, f"Guadagni Annuali - {selected_investor}")
                    gains_series = pd.to_numeric(annual_df['Net_Gain'], errors='coerce').fillna(0.0)
                    best_idx = int(gains_series.idxmax())
                    worst_idx = int(gains_series.idxmin())
                    best_year = int(annual_df.loc[best_idx, 'Year'])
                    best_gain = float(gains_series.iloc[best_idx])
                    best_roi_val = annual_df.loc[best_idx, 'ROI %'] if 'ROI %' in annual_df.columns else None
                    best_roi = float(best_roi_val) if (best_roi_val is not None and pd.notna(best_roi_val)) else 0.0
                    worst_year = int(annual_df.loc[worst_idx, 'Year'])
                    worst_gain = float(gains_series.iloc[worst_idx])
                    worst_roi_val = annual_df.loc[worst_idx, 'ROI %'] if 'ROI %' in annual_df.columns else None
                    worst_roi = float(worst_roi_val) if (worst_roi_val is not None and pd.notna(worst_roi_val)) else 0.0
                    best_color = COA_COLORS['primary_blue'] if best_gain >= 0 else COA_COLORS['primary_purple']
                    worst_color = COA_COLORS['primary_blue'] if worst_gain >= 0 else COA_COLORS['primary_purple']
                    colm1, colm2, colm3, colm4 = st.columns(4)
                    with colm1:
                        st.markdown(f"""
                        <div style="background: var(--card-bg); padding: 1rem; border-radius: 8px;">
                            <div style="font-size:0.85rem; color: var(--text-secondary);">Miglior Anno</div>
                            <div style="display:flex; align-items:center; justify-content:space-between; margin-top:0.2rem;">
                                <span style="font-size:1.6rem; color: var(--text-primary);">{best_year}</span>
                                <span style="display:inline-flex; gap:8px; align-items:center;">
                                    <span style="font-size:1.0rem; padding:0.25rem 0.6rem; border-radius:12px; background:{best_color}; color:white;">${best_gain:,.0f}</span>
                                    <span style="font-size:1.0rem; padding:0.25rem 0.6rem; border-radius:12px; background:{best_color}; color:white;">{(best_roi or 0):+.1f}%</span>
                                </span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    with colm2:
                        st.markdown(f"""
                        <div style="background: var(--card-bg); padding: 1rem; border-radius: 8px;">
                            <div style="font-size:0.85rem; color: var(--text-secondary);">Peggior Anno</div>
                            <div style="display:flex; align-items:center; justify-content:space-between; margin-top:0.2rem;">
                                <span style="font-size:1.6rem; color: var(--text-primary);">{worst_year}</span>
                                <span style="display:inline-flex; gap:8px; align-items:center;">
                                    <span style="font-size:1.0rem; padding:0.25rem 0.6rem; border-radius:12px; background:{worst_color}; color:white;">${worst_gain:,.0f}</span>
                                    <span style="font-size:1.0rem; padding:0.25rem 0.6rem; border-radius:12px; background:{worst_color}; color:white;">{(worst_roi or 0):+.1f}%</span>
                                </span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    with colm3:
                        avg_gain = float(gains_series.mean() or 0.0)
                        final_end_value = float(pd.Series(annual_df['End_Value']).dropna().iloc[-1] if not pd.Series(annual_df['End_Value']).dropna().empty else 0.0)
                        total_deposits_all = float(pd.Series(annual_df['Deposits']).sum() or 0.0)
                        total_roi_pct = ((final_end_value - total_deposits_all) / total_deposits_all * 100) if total_deposits_all > 0 else 0.0
                        years_count = int(pd.Series(annual_df['Year']).nunique()) if 'Year' in annual_df.columns else len(annual_df)
                        avg_roi = float(total_roi_pct / years_count) if years_count > 0 else 0.0
                        st.markdown(f"""
                        <div style="background: var(--card-bg); padding: 1rem; border-radius: 8px;">
                            <div style="font-size:0.85rem; color: var(--text-secondary);">Media Annua</div>
                            <div style="display:flex; gap:8px; align-items:center;">
                                <span style="font-size:1.6rem; color: var(--text-primary);">${avg_gain:,.0f}</span>
                                <span style="padding:0.25rem 0.6rem; border-radius:12px; background:{COA_COLORS['primary_blue'] if avg_roi >= 0 else COA_COLORS['primary_purple']}; color:white;">{avg_roi:+.1f}%</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    with colm4:
                        total_gain = float(gains_series.sum() or 0.0)
                        final_end_value = float(pd.Series(annual_df['End_Value']).dropna().iloc[-1] if not pd.Series(annual_df['End_Value']).dropna().empty else 0.0)
                        total_deposits_all = float(pd.Series(annual_df['Deposits']).sum() or 0.0)
                        total_roi = ((final_end_value - total_deposits_all) / total_deposits_all * 100) if total_deposits_all > 0 else 0.0
                        st.markdown(f"""
                        <div style="background: var(--card-bg); padding: 1rem; border-radius: 8px;">
                            <div style="font-size:0.85rem; color: var(--text-secondary);">Guadagno Totale</div>
                            <div style="display:flex; gap:8px; align-items:center;">
                                <span style="font-size:1.6rem; color: var(--text-primary);">${total_gain:,.0f}</span>
                                <span style="padding:0.25rem 0.6rem; border-radius:12px; background:{COA_COLORS['primary_blue'] if total_roi >= 0 else COA_COLORS['primary_purple']}; color:white;">{total_roi:+.1f}%</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    st.subheader('💰 Flussi Annuali')
                    flows_df = annual_df[['Year','Deposits','Start_of_Year','Withdrawals','End_Value','ROI %']].copy()
                    flows_df.rename(columns={
                        'Year': 'Anno',
                        'Deposits': 'Depositi',
                        'Start_of_Year': 'Inizio Anno',
                        'Withdrawals': 'Prelievi',
                        'End_Value': 'Valore Fine Anno',
                        'ROI %': 'ROI %'
                    }, inplace=True)
                    flows_df = flows_df[['Anno','Depositi','Inizio Anno','Prelievi','Valore Fine Anno','ROI %']]
                    flows_df['Anno'] = flows_df['Anno'].astype(int)
                    fmt_cols = ['Inizio Anno','Depositi','Prelievi','Valore Fine Anno','ROI %']
                    for c in fmt_cols:
                        flows_df[c] = pd.to_numeric(flows_df[c], errors='coerce').fillna(0.0)
                    st.dataframe(
                        flows_df.style.format({'Inizio Anno': '${:,.0f}', 'Depositi': '${:,.0f}', 'Prelievi': '${:,.0f}', 'Valore Fine Anno': '${:,.0f}', 'ROI %': '{:+.1f}%'}),
                        use_container_width=True,
                        hide_index=True
                    )
                    st.subheader('📆 Guadagni Mensili')
                    inv_events = all_events_df[all_events_df['investor'] == selected_investor]
                    min_year = datetime.date.today().year
                    if not inv_events.empty:
                        min_year = int(pd.to_datetime(inv_events['date']).dt.year.min())
                    
                    curr_year = datetime.date.today().year
                    # Ensure current year is always included and range is valid
                    start_y = min(min_year, curr_year)
                    year_options = list(range(start_y, curr_year + 1))

                    
                    if year_options:
                        selected_year = st.selectbox('Anno', options=year_options, index=len(year_options)-1, key=f"monthly_year_admin_{selected_investor}")
                        monthly_df = calculate_monthly_performance(selected_investor, selected_year, all_events_df, total_history)
                        if monthly_df is not None and not monthly_df.empty:
                            display_monthly_chart(monthly_df, f"Guadagni Mensili - {selected_investor} ({selected_year})")
                        else:
                            st.info("Nessun dato mensile disponibile per l'anno selezionato.")
                    else:
                        st.info("Nessun dato disponibile per i guadagni mensili.")
        else:
            if user_investor_name:
                st.subheader('📈 Il Tuo Storico Annuale')
                annual_df = calculate_annual_performance(user_investor_name, all_events_df, total_history)
                if annual_df is not None and not annual_df.empty:
                    inv_norm_u = user_investor_name.strip().lower() if user_investor_name else ''
                    th_cols_norm_u = {str(c).strip().lower(): c for c in total_history.columns if c not in ['date','total']} if (total_history is not None and not total_history.empty) else {}
                    th_col_u = th_cols_norm_u.get(inv_norm_u)
                    if total_history is not None and not total_history.empty and th_col_u is not None:
                        ph_fix = total_history.copy()
                        ph_fix['date'] = pd.to_datetime(ph_fix['date']).dt.date
                        for i, r in annual_df.iterrows():
                            y = int(r['Year'])
                            end_dt = datetime.date(y, 12, 31)
                            upto_end = ph_fix[ph_fix['date'] <= end_dt]
                            if not upto_end.empty:
                                end_bal_fix = float(upto_end[th_col_u].iloc[-1])
                            else:
                                events_upto_end = all_events_df[pd.to_datetime(all_events_df['date']).dt.date <= end_dt]
                                end_balances, _ = replay_events(events_upto_end)
                                kb_map = {str(k).strip().lower(): k for k in end_balances.keys()}
                                end_bal_fix = float(end_balances.get(kb_map.get(inv_norm_u, user_investor_name), 0.0))
                            ev = all_events_df.copy()
                            ev['__inv_norm__'] = ev['investor'].astype(str).str.strip().str.lower()
                            year_mask = (pd.to_datetime(ev['date']).dt.year == y) & (ev['__inv_norm__'] == inv_norm_u)
                            deposits_fix = float(all_events_df[year_mask & (all_events_df['type'] == 'deposit')]['usd_amount'].sum() or 0.0)
                            withdrawals_fix = float(all_events_df[year_mask & (all_events_df['type'] == 'withdrawal')]['usd_amount'].sum() or 0.0)
                            start_bal_fix = float(r['Start_Year_Balance'] or 0.0)
                            start_of_year_fix = start_bal_fix + deposits_fix
                            net_gain_fix = end_bal_fix - start_bal_fix - deposits_fix + withdrawals_fix
                            roi_fix = float(net_gain_fix / start_of_year_fix * 100) if start_of_year_fix > 0 else None
                            annual_df.at[i, 'End_Value'] = float(end_bal_fix)
                            annual_df.at[i, 'Start_of_Year'] = float(start_of_year_fix)
                            annual_df.at[i, 'Deposits'] = float(deposits_fix)
                            annual_df.at[i, 'Withdrawals'] = float(withdrawals_fix)
                            annual_df.at[i, 'Net_Gain'] = float(net_gain_fix)
                            annual_df.at[i, 'ROI %'] = roi_fix
                    annual_df['Start_of_Year'] = pd.to_numeric(annual_df.get('Start_Year_Balance', 0.0), errors='coerce').fillna(0.0) + pd.to_numeric(annual_df.get('Deposits', 0.0), errors='coerce').fillna(0.0)
                    annual_df['Net_Gain'] = (
                        pd.to_numeric(annual_df.get('End_Value', 0.0), errors='coerce').fillna(0.0)
                        - pd.to_numeric(annual_df.get('Start_Year_Balance', annual_df.get('Start_Value', 0.0)), errors='coerce').fillna(0.0)
                        - pd.to_numeric(annual_df.get('Deposits', 0.0), errors='coerce').fillna(0.0)
                        + pd.to_numeric(annual_df.get('Withdrawals', 0.0), errors='coerce').fillna(0.0)
                    )
                    annual_df['ROI %'] = annual_df.apply(lambda r: (float(r['Net_Gain']) / float(r['Start_of_Year']) * 100) if float(r['Start_of_Year']) > 0 else None, axis=1)
                    display_annual_chart(annual_df, 'La Tua Performance')
                    gains_series_u = pd.to_numeric(annual_df['Net_Gain'], errors='coerce').fillna(0.0)
                    best_idx_u = int(gains_series_u.idxmax())
                    worst_idx_u = int(gains_series_u.idxmin())
                    best_year_u = int(annual_df.loc[best_idx_u, 'Year'])
                    best_gain_u = float(gains_series_u.iloc[best_idx_u])
                    best_roi_val_u = annual_df.loc[best_idx_u, 'ROI %'] if 'ROI %' in annual_df.columns else None
                    best_roi_u = float(best_roi_val_u) if (best_roi_val_u is not None and pd.notna(best_roi_val_u)) else 0.0
                    worst_year_u = int(annual_df.loc[worst_idx_u, 'Year'])
                    worst_gain_u = float(gains_series_u.iloc[worst_idx_u])
                    worst_roi_val_u = annual_df.loc[worst_idx_u, 'ROI %'] if 'ROI %' in annual_df.columns else None
                    worst_roi_u = float(worst_roi_val_u) if (worst_roi_val_u is not None and pd.notna(worst_roi_val_u)) else 0.0
                    best_color_u = COA_COLORS['primary_blue'] if best_gain_u >= 0 else COA_COLORS['primary_purple']
                    worst_color_u = COA_COLORS['primary_blue'] if worst_gain_u >= 0 else COA_COLORS['primary_purple']
                    colu1, colu2, colu3, colu4 = st.columns(4)
                    with colu1:
                        st.markdown(f"""
                        <div style="background: var(--card-bg); padding: 1rem; border-radius: 8px;">
                            <div style="font-size:0.85rem; color: var(--text-secondary);">Miglior Anno</div>
                            <div style="display:flex; align-items:center; justify-content:space-between; margin-top:0.2rem;">
                                <span style="font-size:1.6rem; color: var(--text-primary);">{best_year_u}</span>
                                <span style="display:inline-flex; gap:8px; align-items:center;">
                                    <span style="font-size:1.0rem; padding:0.25rem 0.6rem; border-radius:12px; background:{best_color_u}; color:white;">${best_gain_u:,.0f}</span>
                                    <span style="font-size:1.0rem; padding:0.25rem 0.6rem; border-radius:12px; background:{best_color_u}; color:white;">{(best_roi_u or 0):+.1f}%</span>
                                </span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    with colu2:
                        st.markdown(f"""
                        <div style="background: var(--card-bg); padding: 1rem; border-radius: 8px;">
                            <div style="font-size:0.85rem; color: var(--text-secondary);">Peggior Anno</div>
                            <div style="display:flex; align-items:center; justify-content:space-between; margin-top:0.2rem;">
                                <span style="font-size:1.6rem; color: var(--text-primary);">{worst_year_u}</span>
                                <span style="display:inline-flex; gap:8px; align-items:center;">
                                    <span style="font-size:1.0rem; padding:0.25rem 0.6rem; border-radius:12px; background:{worst_color_u}; color:white;">${worst_gain_u:,.0f}</span>
                                    <span style="font-size:1.0rem; padding:0.25rem 0.6rem; border-radius:12px; background:{worst_color_u}; color:white;">{(worst_roi_u or 0):+.1f}%</span>
                                </span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    with colu3:
                        avg_gain_u = float(gains_series_u.mean() or 0.0)
                        final_end_value_u = float(pd.Series(annual_df['End_Value']).dropna().iloc[-1] if not pd.Series(annual_df['End_Value']).dropna().empty else 0.0)
                        total_deposits_u = float(pd.Series(annual_df['Deposits']).sum() or 0.0)
                        total_roi_pct_u = ((final_end_value_u - total_deposits_u) / total_deposits_u * 100) if total_deposits_u > 0 else 0.0
                        years_count_u = int(pd.Series(annual_df['Year']).nunique()) if 'Year' in annual_df.columns else len(annual_df)
                        avg_roi_u = float(total_roi_pct_u / years_count_u) if years_count_u > 0 else 0.0
                        st.markdown(f"""
                        <div style="background: var(--card-bg); padding: 1rem; border-radius: 8px;">
                            <div style="font-size:0.85rem; color: var(--text-secondary);">Media Annua</div>
                            <div style="display:flex; gap:8px; align-items:center;">
                                <span style="font-size:1.6rem; color: var(--text-primary);">${avg_gain_u:,.0f}</span>
                                <span style="padding:0.25rem 0.6rem; border-radius:12px; background:{COA_COLORS['primary_blue'] if avg_roi_u >= 0 else COA_COLORS['primary_purple']}; color:white;">{avg_roi_u:+.1f}%</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    with colu4:
                        total_gain_u = float(gains_series_u.sum() or 0.0)
                        final_end_value_u = float(pd.Series(annual_df['End_Value']).dropna().iloc[-1] if not pd.Series(annual_df['End_Value']).dropna().empty else 0.0)
                        total_deposits_u2 = float(pd.Series(annual_df['Deposits']).sum() or 0.0)
                        total_roi_u = ((final_end_value_u - total_deposits_u2) / total_deposits_u2 * 100) if total_deposits_u2 > 0 else 0.0
                        st.markdown(f"""
                        <div style="background: var(--card-bg); padding: 1rem; border-radius: 8px;">
                            <div style="font-size:0.85rem; color: var(--text-secondary);">Guadagno Totale</div>
                            <div style="display:flex; gap:8px; align-items:center;">
                                <span style="font-size:1.6rem; color: var(--text-primary);">${total_gain_u:,.0f}</span>
                                <span style="padding:0.25rem 0.6rem; border-radius:12px; background:{COA_COLORS['primary_blue'] if total_roi_u >= 0 else COA_COLORS['primary_purple']}; color:white;">{total_roi_u:+.1f}%</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    st.subheader('💰 I Tuoi Flussi Annuali')
                    flows_df = annual_df[['Year','Deposits','Start_of_Year','Withdrawals','End_Value','ROI %']].copy()
                    flows_df.rename(columns={
                        'Year': 'Anno',
                        'Deposits': 'Depositi',
                        'Start_of_Year': 'Inizio Anno',
                        'Withdrawals': 'Prelievi',
                        'End_Value': 'Valore Fine Anno',
                        'ROI %': 'ROI %'
                    }, inplace=True)
                    flows_df = flows_df[['Anno','Depositi','Inizio Anno','Prelievi','Valore Fine Anno','ROI %']]
                    flows_df['Anno'] = flows_df['Anno'].astype(int)
                    fmt_cols = ['Inizio Anno','Depositi','Prelievi','Valore Fine Anno','ROI %']
                    for c in fmt_cols:
                        flows_df[c] = pd.to_numeric(flows_df[c], errors='coerce').fillna(0.0)
                    st.dataframe(
                        flows_df.style.format({'Inizio Anno': '${:,.0f}', 'Depositi': '${:,.0f}', 'Prelievi': '${:,.0f}', 'Valore Fine Anno': '${:,.0f}', 'ROI %': '{:+.1f}%'}),
                        use_container_width=True,
                        hide_index=True
                    )
                    st.subheader('📆 Guadagni Mensili')
                    inv_events_u = all_events_df[all_events_df['investor'] == user_investor_name]
                    min_year_u = datetime.date.today().year
                    if not inv_events_u.empty:
                        min_year_u = int(pd.to_datetime(inv_events_u['date']).dt.year.min())
                    
                    curr_year_u = datetime.date.today().year
                    # Ensure current year is always included and range is valid
                    start_y_u = min(min_year_u, curr_year_u)
                    year_options_u = list(range(start_y_u, curr_year_u + 1))

                    if year_options_u:
                        selected_year_u = st.selectbox('Anno', options=year_options_u, index=len(year_options_u)-1, key="monthly_year_user")
                        monthly_df_u = calculate_monthly_performance(user_investor_name, selected_year_u, all_events_df, total_history)
                        if monthly_df_u is not None and not monthly_df_u.empty:
                            display_monthly_chart(monthly_df_u, f"Guadagni Mensili - {user_investor_name} ({selected_year_u})")
                        else:
                            st.info("Nessun dato mensile disponibile per l'anno selezionato.")
                    else:
                        st.info("Nessun dato disponibile per i guadagni mensili.")

    # Aggiornamenti Tab
    with tabs[2]:
        st.markdown("### 📝 Aggiornamenti")
        if 'updates_df' in locals():
            df_u = updates_df.copy()
        else:
            with get_db_session() as db:
                df_u = pd.read_sql(db.query(UpdateNote).statement, db.bind)
        if df_u.empty:
            st.info("Nessun aggiornamento disponibile")
        else:
            df_u['year'] = pd.to_numeric(df_u['year'], errors='coerce').fillna(0).astype(int)
            df_u['month'] = pd.to_numeric(df_u['month'], errors='coerce').fillna(0).astype(int)
            years = sorted([y for y in df_u['year'].unique() if y > 0], reverse=True)
            selected_year = st.selectbox('Anno', options=years) if years else None
            if selected_year is not None:
                df_year = df_u[df_u['year'] == selected_year]
                months = sorted([m for m in df_year['month'].unique() if 1 <= m <= 12])
                month_names = {1:'Gennaio',2:'Febbraio',3:'Marzo',4:'Aprile',5:'Maggio',6:'Giugno',7:'Luglio',8:'Agosto',9:'Settembre',10:'Ottobre',11:'Novembre',12:'Dicembre'}
                month_labels = [month_names.get(m, str(m)) for m in months]
                month_tabs = st.tabs(month_labels) if months else []
                for i, m in enumerate(months):
                    with month_tabs[i]:
                        df_m = df_year[df_year['month'] == m].copy()
                        strategy_mapping = strategies_df[['id','name']].set_index('id')['name'].to_dict() if strategies_df is not None and not strategies_df.empty else {}
                        df_m['protocol_name'] = df_m['strategy_id'].map(strategy_mapping).fillna('Nota Generale')
                        groups = df_m.groupby('protocol_name', dropna=False)
                        protos = list(groups.groups.keys())
                        if protos:
                            cols = st.columns(len(protos))
                            for j, proto in enumerate(protos):
                                with cols[j]:
                                    st.markdown(f"<div style='font-size:1.25rem; font-weight:700; color: var(--text-primary); margin-bottom: 0.5rem;'>{proto}</div>", unsafe_allow_html=True)
                                    group_rows = groups.get_group(proto).sort_values('id')
                                    items_html = "".join(
                                        f"<li style='white-space: pre-line;'>{str(r['note']) if r['note'] is not None else ''}</li>"
                                        for _, r in group_rows.iterrows()
                                    )
                                    st.markdown(f"<ul style='margin:0; padding-left:1.2rem;'>{items_html}</ul>", unsafe_allow_html=True)
                        else:
                            st.info("Nessuna nota per questo mese")

        if current_role == 'admin':
            st.divider()
            st.subheader("➕ Aggiungi Nota")
            with st.form("add_update_note_form"):
                now = datetime.date.today()
                a_year = st.number_input('Anno', min_value=2000, max_value=2100, value=now.year, step=1)
                month_map = {1:'Gennaio',2:'Febbraio',3:'Marzo',4:'Aprile',5:'Maggio',6:'Giugno',7:'Luglio',8:'Agosto',9:'Settembre',10:'Ottobre',11:'Novembre',12:'Dicembre'}
                a_month = st.selectbox('Mese', options=list(range(1,13)), index=now.month-1, format_func=lambda m: month_map[m])
                proto_options = get_protocol_options(strategies_df, include_no=True)
                selected_proto = st.selectbox('Protocollo', options=proto_options)
                note_text = st.text_area('Note', placeholder='Inserisci le note...')
                if st.form_submit_button('Salva Nota', use_container_width=True):
                    with get_db_session() as db:
                        strategy_id = resolve_strategy_id_by_name(strategies_df, selected_proto) if selected_proto != 'No Protocol' else None
                        db.add(UpdateNote(year=int(a_year), month=int(a_month), strategy_id=strategy_id, note=note_text.strip(), created_by=current_user, created_at=datetime.date.today()))
                    st.success("Nota salvata")
                    time.sleep(1)
                    st.rerun()

            st.subheader("✏️ Modifica/Elimina Nota")
            with get_db_session() as db:
                notes_to_edit = pd.read_sql(db.query(UpdateNote).statement, db.bind)
            if not notes_to_edit.empty:
                strategy_mapping = strategies_df[['id','name']].set_index('id')['name'].to_dict() if strategies_df is not None and not strategies_df.empty else {}
                notes_to_edit['protocol_name'] = notes_to_edit['strategy_id'].map(strategy_mapping).fillna('Nota Generale')
                notes_to_edit = notes_to_edit[['id','year','month','protocol_name','note','created_by','created_at']].sort_values(['year','month','id'], ascending=[False, False, False])
                st.dataframe(notes_to_edit, use_container_width=True, hide_index=True)
                sel_id = st.selectbox('Seleziona ID Nota', options=sorted(notes_to_edit['id'].tolist(), reverse=True))
                if sel_id:
                    with get_db_session() as db:
                        n_row = db.get(UpdateNote, int(sel_id))
                        if n_row:
                            with st.form(f"edit_update_note_{n_row.id}"):
                                e_year = st.number_input('Anno', min_value=2000, max_value=2100, value=int(n_row.year), step=1)
                                month_map = {1:'Gennaio',2:'Febbraio',3:'Marzo',4:'Aprile',5:'Maggio',6:'Giugno',7:'Luglio',8:'Agosto',9:'Settembre',10:'Ottobre',11:'Novembre',12:'Dicembre'}
                                e_month = st.selectbox('Mese', options=list(range(1,13)), index=max(0, int(n_row.month)-1), format_func=lambda m: month_map[m])
                                current_protocol = 'No Protocol'
                                if n_row.strategy_id and strategy_mapping:
                                    current_protocol = strategy_mapping.get(int(n_row.strategy_id), 'No Protocol')
                                proto_options = get_protocol_options(strategies_df, include_no=True)
                                e_protocol = st.selectbox('Protocollo', options=proto_options, index=proto_options.index(current_protocol) if current_protocol in proto_options else 0)
                                e_note = st.text_area('Note', value=n_row.note or '')
                                c1, c2 = st.columns(2)
                                with c1:
                                    if st.form_submit_button('💾 Aggiorna', use_container_width=True):
                                        n_row.year = int(e_year)
                                        n_row.month = int(e_month)
                                        if e_protocol != 'No Protocol':
                                            n_row.strategy_id = resolve_strategy_id_by_name(strategies_df, e_protocol)
                                        else:
                                            n_row.strategy_id = None
                                        n_row.note = (e_note or '').strip()
                                        db.commit()
                                        st.success(f"Nota #{n_row.id} aggiornata")
                                        time.sleep(1)
                                        st.rerun()
                                with c2:
                                    if st.form_submit_button('🗑️ Elimina', type='primary', use_container_width=True):
                                        db.delete(n_row)
                                        db.commit()
                                        st.warning(f"Nota #{n_row.id} eliminata")
                                        time.sleep(1)
                                        st.rerun()

    # Event Management Tab (Admin Only)
    if current_role == 'admin':
        with tabs[-1] if len(tabs) > 0 else st.container():
            st.markdown("### ⚙️ Gestione Eventi")
            
            form_cols = st.columns(2)
            
            with form_cols[0]:
                st.subheader("➕ Aggiungi Evento")
                
                tab_dep, tab_wd, tab_val = st.tabs(["💰 Deposito", "💸 Prelievo", "📈 Valutazione"])
                
                with tab_dep:
                    with st.form("deposit_form"):
                        d_date = st.date_input('Data', datetime.date.today(), key='d_date')
                        d_investor = st.text_input('Nome Investitore', placeholder='Inserisci nome investitore', key='d_inv')
                        
                        d_protocol = st.selectbox('Protocollo', get_protocol_options(strategies_df, include_no=True), key='d_protocol')
                        
                        d_eur = st.number_input('Importo (EUR)', min_value=0.01, step=100.0, key='d_eur')
                        
                        if st.form_submit_button('💰 Aggiungi Deposito', use_container_width=True):
                            rate = get_historical_eurusd(d_date)
                            usd_amount = d_eur * rate
                            
                            with get_db_session() as db:
                                strategy_id = resolve_strategy_id_by_name(strategies_df, d_protocol) if d_protocol != 'No Protocol' else None
                                
                                db.add(Event(
                                    date=d_date, 
                                    type='deposit', 
                                    strategy_id=strategy_id,
                                    investor=d_investor.strip(), 
                                    eur_amount=d_eur, 
                                    usd_amount=usd_amount, 
                                    eurusd_rate=rate
                                ))
                            
                            st.success('Deposito aggiunto con successo!')
                            time.sleep(1)
                            st.rerun()
                
                with tab_wd:
                    with st.form("withdrawal_form"):
                        w_date = st.date_input('Data', datetime.date.today(), key='w_date')
                        w_investor = st.text_input('Nome Investitore', placeholder='Inserisci nome investitore', key='w_inv')
                        
                        w_protocol = st.selectbox('Protocollo', get_protocol_options(strategies_df, include_no=True), key='w_protocol')
                        
                        w_usd = st.number_input('Importo (USD)', min_value=0.01, step=100.0, key='w_usd')
                        
                        if st.form_submit_button('💸 Aggiungi Prelievo', use_container_width=True):
                            rate = get_historical_eurusd(w_date)
                            eur_amount = w_usd / rate
                            
                            with get_db_session() as db:
                                strategy_id = resolve_strategy_id_by_name(strategies_df, w_protocol) if w_protocol != 'No Protocol' else None
                                
                                db.add(Event(
                                    date=w_date, 
                                    type='withdrawal', 
                                    strategy_id=strategy_id,
                                    investor=w_investor.strip(), 
                                    eur_amount=eur_amount, 
                                    usd_amount=w_usd, 
                                    eurusd_rate=rate
                                ))
                            
                            st.success('Prelievo aggiunto con successo!')
                            time.sleep(1)
                            st.rerun()
                
                with tab_val:
                    with st.form("valuation_form"):
                        v_date = st.date_input('Data', datetime.date.today(), key='v_date')
                        
                        v_protocol = st.selectbox('Protocollo', get_protocol_options(strategies_df, include_all=True), key='v_protocol')
                        
                        v_total = st.number_input('Valore Totale Portafoglio (USD)', min_value=0.01, step=1000.0, key='v_usd')
                        
                        if st.form_submit_button('📈 Aggiungi Valutazione', use_container_width=True):
                            with get_db_session() as db:
                                strategy_id = resolve_strategy_id_by_name(strategies_df, v_protocol) if v_protocol != 'All Protocols' else None
                                
                                db.add(Event(
                                    date=v_date, 
                                    type='valuation', 
                                    strategy_id=strategy_id,
                                    valuation_total_usd=v_total
                                ))
                            
                            st.success('Valutazione aggiunta con successo!')
                            time.sleep(1)
                            st.rerun()
            
            with form_cols[1]:
                st.subheader("✏️ Modifica/Elimina Eventi")
                
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
                        
                        ev_id_to_edit = st.selectbox('Seleziona ID Evento da modificare', options=sorted(events_to_edit['id'].tolist(), reverse=True))
                        
                        if ev_id_to_edit:
                            ev_row = db.get(Event, ev_id_to_edit)
                            if ev_row:
                                with st.form(f"edit_form_{ev_row.id}"):
                                    st.markdown(f"**Modifica Evento #{ev_row.id} ({ev_row.type})**")
                                    
                                    e_date = st.date_input('Data', value=ev_row.date)
                                    
                                    if ev_row.type in ['deposit', 'withdrawal']:
                                        e_investor = st.text_input('Nome Investitore', value=ev_row.investor or "")
                                        
                                        current_protocol = 'No Protocol'
                                        if ev_row.strategy_id and not strategies_df.empty:
                                            match = strategies_df[strategies_df['id'] == ev_row.strategy_id]
                                            if not match.empty:
                                                current_protocol = match['name'].iloc[0]
                                        protocol_options = get_protocol_options(strategies_df, include_no=True)
                                        e_protocol = st.selectbox('Protocollo', protocol_options, index=protocol_options.index(current_protocol))
                                    
                                    if ev_row.type == 'deposit':
                                        e_eur = st.number_input('Importo (EUR)', value=ev_row.eur_amount or 0.0)
                                    elif ev_row.type == 'withdrawal':
                                        e_usd = st.number_input('Importo (USD)', value=ev_row.usd_amount or 0.0)
                                    else:  # valuation
                                        e_val = st.number_input('Valore Totale (USD)', value=ev_row.valuation_total_usd or 0.0)
                                    
                                    col_btn1, col_btn2 = st.columns(2)
                                    
                                    with col_btn1:
                                        if st.form_submit_button('💾 Aggiorna', use_container_width=True):
                                            ev_row.date = e_date
                                            rate = get_historical_eurusd(e_date)
                                            
                                            if ev_row.type in ['deposit', 'withdrawal']:
                                                ev_row.investor = e_investor.strip() or None
                                                
                                                if e_protocol != 'No Protocol':
                                                    ev_row.strategy_id = resolve_strategy_id_by_name(strategies_df, e_protocol)
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
                                            st.success(f"Evento #{ev_row.id} aggiornato!")
                                            time.sleep(1)
                                            st.rerun()
                                    
                                    with col_btn2:
                                        if st.form_submit_button('🗑️ Elimina', type="primary", use_container_width=True):
                                            db.delete(ev_row)
                                            db.commit()
                                            st.success(f"Evento #{ev_row.id} eliminato!")
                                            time.sleep(1)
                                            st.rerun()
                    else:
                        st.info("Nessun evento da modificare")

# ---------- Admin Panel ----------
if current_role == 'admin':
    st.markdown("### 👑 Pannello Amministratore:")
    with st.expander('Gestione Utenti'):
        admin_cols = st.columns(2)
        
        with admin_cols[0]:
            st.subheader('➕ Crea Nuovo Utente')
            with st.form("create_user_form"):
                new_username = st.text_input('Nome Utente', placeholder='Inserisci nome utente')
                new_password = st.text_input('Password', type='password', placeholder='Inserisci password')
                new_role = st.selectbox('Ruolo', ['user', 'admin'])
                new_investor_name = st.text_input('Nome Investitore (opzionale)', placeholder='Collega a un investitore')
                
                if st.form_submit_button('Crea Utente', use_container_width=True):
                    if new_username and new_password:
                        try:
                            with get_db_session() as db:
                                if db.query(User).filter(User.username == new_username).first():
                                    st.error('Nome utente già esistente')
                                else:
                                    db.add(User(
                                        username=new_username.strip(), 
                                        password_hash=hash_password(new_password), 
                                        role=new_role, 
                                        investor_name=new_investor_name.strip() or None
                                    ))
                                    db.commit()  # Commit the transaction
                                    st.success(f'Utente "{new_username}" creato con successo!')
                                    time.sleep(1)
                                    st.rerun()
                        except Exception as e:
                            st.error(f"Errore Database: {e}")
                    else:
                        st.warning('Nome utente e password sono richiesti')
        
        with admin_cols[1]:
            st.subheader('✏️ Modifica/Elimina Utente')
            with get_db_session() as db:
                all_users = db.query(User).all()
                
                if all_users:
                    user_options = [f"{user.username} ({user.role})" for user in all_users]
                    selected_user = st.selectbox("Seleziona utente", options=user_options)
                    
                    if selected_user:
                        username = selected_user.split(' (')[0]
                        user_to_edit = db.query(User).filter(User.username == username).first()
                        
                        if user_to_edit:
                            with st.form("edit_user_form"):
                                is_admin_user = (user_to_edit.username == 'admin')
                                
                                st.markdown(f"**Gestione Utente: {user_to_edit.username}**")
                                
                                edited_username = st.text_input(
                                    "Nome Utente", 
                                    value=user_to_edit.username, 
                                    disabled=is_admin_user
                                )
                                
                                edited_role = st.selectbox(
                                    "Ruolo", 
                                    ['user', 'admin'], 
                                    index=1 if user_to_edit.role == 'admin' else 0, 
                                    disabled=is_admin_user
                                )
                                
                                edited_inv_name = st.text_input(
                                    "Nome Investitore", 
                                    value=user_to_edit.investor_name or "",
                                    disabled=is_admin_user
                                )
                                
                                new_password = st.text_input(
                                    "Nuova Password (lascia vuoto per mantenere corrente)", 
                                    type="password"
                                )
                                
                                col_btn1, col_btn2 = st.columns(2)
                                
                                with col_btn1:
                                    if st.form_submit_button("💾 Aggiorna Utente", use_container_width=True):
                                        if not is_admin_user:
                                            user_to_edit.investor_name = edited_inv_name.strip() or None
                                            
                                            if edited_username != user_to_edit.username:
                                                existing = db.query(User).filter(User.username == edited_username).first()
                                                if existing:
                                                    st.error("Nome utente già in uso")
                                                else:
                                                    user_to_edit.username = edited_username
                                            
                                            user_to_edit.role = edited_role
                                        
                                        if new_password:
                                            user_to_edit.password_hash = hash_password(new_password)
                                        
                                        db.commit()
                                        st.success(f"Utente '{user_to_edit.username}' aggiornato!")
                                        time.sleep(1)
                                        st.rerun()
                                
                                with col_btn2:
                                    if not is_admin_user and st.form_submit_button(
                                        "🗑️ ELIMINA UTENTE", 
                                        type="primary", 
                                        use_container_width=True
                                    ):
                                        db.delete(user_to_edit)
                                        db.commit()
                                        st.warning(f"Utente '{user_to_edit.username}' eliminato!")
                                        time.sleep(1)
                                        st.rerun()
                else:
                    st.info("Nessun utente da modificare")
        
        st.divider()
        st.subheader('👥 Utenti Correnti')
        with get_db_session() as db:
            users_df = pd.read_sql(db.query(User).statement, db.bind)
            if not users_df.empty:
                display_df = users_df[['id', 'username', 'role', 'investor_name']].copy()
                st.dataframe(display_df, use_container_width=True, hide_index=True)
            else:
                st.info("Nessun utente trovato")

    with st.expander('Gestione Protocolli'):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader('➕ Aggiungi Nuovo Protocollo')
            with st.form("add_protocol_form"):
                protocol_name = st.text_input('Nome Protocollo', placeholder='es., Growth Protocol')
                protocol_desc = st.text_area('Descrizione', placeholder='Descrivi il protocollo...')
                
                if st.form_submit_button('Aggiungi Protocollo', use_container_width=True):
                    if protocol_name:
                        with get_db_session() as db:
                            existing = db.query(Strategy).filter(Strategy.name == protocol_name).first()
                            if existing:
                                st.error('Nome protocollo già esistente')
                            else:
                                new_protocol = Strategy(name=protocol_name, description=protocol_desc)
                                db.add(new_protocol)
                                db.commit()
                                st.success(f'Protocollo "{protocol_name}" aggiunto con successo!')
                                time.sleep(1)
                                st.rerun()
                    else:
                        st.error('Il nome del protocollo è richiesto')
        
        with col2:
            st.subheader('📋 Protocolli Attivi')
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
                                if st.button('✏️', key=f'rename_protocol_{protocol.id}', help='Rinomina Protocollo'):
                                    st.session_state[f'renaming_protocol_{protocol.id}'] = True
                                    st.session_state[f'rename_name_{protocol.id}'] = protocol.name
                                    st.session_state[f'rename_desc_{protocol.id}'] = protocol.description or ''
                            with col_c:
                                if st.button('🗑️', key=f'del_protocol_{protocol.id}', help='Elimina Protocollo'):
                                    protocol_obj = db.get(Strategy, protocol.id)
                                    protocol_obj.is_active = False
                                    db.commit()
                                    st.success('Protocollo disattivato')
                                    time.sleep(1)
                                    st.rerun()
                            
                            if st.session_state.get(f'renaming_protocol_{protocol.id}', False):
                                with st.form(f'rename_form_{protocol.id}'):
                                    new_name = st.text_input('Nuovo Nome', 
                                                             value=st.session_state[f'rename_name_{protocol.id}'],
                                                             key=f'rename_name_input_{protocol.id}')
                                    new_desc = st.text_area('Nuova Descrizione (opzionale)', 
                                                            value=st.session_state[f'rename_desc_{protocol.id}'],
                                                            key=f'rename_desc_input_{protocol.id}')
                                    
                                    col_rename1, col_rename2 = st.columns(2)
                                    with col_rename1:
                                        if st.form_submit_button('💾 Salva', use_container_width=True):
                                            if new_name and new_name != protocol.name:
                                                existing = db.query(Strategy).filter(Strategy.name == new_name).first()
                                                if existing:
                                                    st.error('Nome protocollo già esistente')
                                                else:
                                                    protocol_obj = db.get(Strategy, protocol.id)
                                                    protocol_obj.name = new_name
                                                    protocol_obj.description = new_desc
                                                    db.commit()
                                                    st.success(f'Protocollo rinominato in "{new_name}"')
                                                    st.session_state[f'renaming_protocol_{protocol.id}'] = False
                                                    time.sleep(1)
                                                    st.rerun()
                                            elif new_name == protocol.name:
                                                protocol_obj = db.get(Strategy, protocol.id)
                                                protocol_obj.description = new_desc
                                                db.commit()
                                                st.success('Descrizione protocollo aggiornata')
                                                st.session_state[f'renaming_protocol_{protocol.id}'] = False
                                                time.sleep(1)
                                                st.rerun()
                                    with col_rename2:
                                        if st.form_submit_button('❌ Annulla', use_container_width=True):
                                            st.session_state[f'renaming_protocol_{protocol.id}'] = False
                                            st.rerun()
                else:
                    st.info('Nessun protocollo ancora definito')

    with st.expander('Gestione Dati'):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader('📤 Esporta Dati')
            if st.button('Esporta tutti gli eventi in CSV', use_container_width=True):
                csv_data = export_events_to_csv()
                st.download_button(
                    label="⬇️ Scarica CSV",
                    data=csv_data,
                    file_name=f"coa_events_export_{datetime.date.today().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        with col2:
            st.subheader('📥 Importa Dati')
            uploaded_file = st.file_uploader("Scegli un file CSV", type="csv")
            if uploaded_file is not None:
                csv_content = uploaded_file.getvalue().decode('utf-8')
                success, message = import_events_from_csv(csv_content)
                if success:
                    st.success(message)
                    time.sleep(2)
                    st.rerun()
                else:
                    st.error(message)
