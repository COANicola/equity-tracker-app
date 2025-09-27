"""
Enhanced Streamlit Equity Tracker - FINAL VERSION 2.7 (User View)

This version implements a completely customized and secure view for standard (non-admin) users.

Key Features:
- Standard users now have a read-only interface with no event management tab.
- User-specific metrics: 'Your Share %' replaces the total investor count.
- Corrected calculations: User's share and value are now calculated against the *total* portfolio, fixing the "100% bug".
- User-specific charts: A new chart shows the user's personal stake value (USD) over time.
- The admin view remains unchanged, retaining full functionality and data visibility.

Run:
pip install streamlit sqlalchemy pandas yfinance passlib plotly pyjwt bcrypt==3.2.0 psycopg2-binary
streamlit run your_script_name.py
"""

import streamlit as st
from sqlalchemy import create_engine, Column, Integer, String, Float, Date, Text
from sqlalchemy.orm import declarative_base, sessionmaker, scoped_session
import pandas as pd
import yfinance as yf
from passlib.context import CryptContext
import datetime
import plotly.express as px
import jwt
from jwt import PyJWTError
import time
from contextlib import contextmanager
import logging

# ... (Tutto il codice di setup, utility e replay_events rimane identico) ...
# ---------- Logging setup ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- Configuration ----------
JWT_SECRET = st.secrets.get('jwt_secret', 'dev-secret-change-me-in-production')
JWT_ALGO = 'HS256'
JWT_EXP_SECONDS = 60 * 60 * 8  # 8 hours

DB_URL = st.secrets.get('db_url', 'sqlite:///equity_app_final.db')

# ---------- DB setup with thread-safe session ----------
Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)
    role = Column(String, nullable=False, default='user')
    investor_name = Column(String, nullable=True)

class Event(Base):
    __tablename__ = 'events'
    id = Column(Integer, primary_key=True)
    date = Column(Date, nullable=False)
    type = Column(String, nullable=False)  # 'deposit', 'withdrawal', 'valuation'
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
    st.warning(f"Could not fetch live exchange rate for {date}. Using an approximate fallback rate.")
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

# ---------- Replay engine & ROI ----------
def replay_events(events_df: pd.DataFrame):
    if events_df.empty: return {}, pd.DataFrame()
    events_df = events_df.sort_values('date').reset_index(drop=True)
    investor_balances, history_rows, prev_total_value = {}, [], 0.0
    for _, row in events_df.iterrows():
        event_type, usd_amount, investor = row.get('type'), float(row.get('usd_amount', 0.0) or 0.0), row.get('investor')
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

# ---------- Streamlit app ----------
st.set_page_config(page_title='Equity Tracker', layout='wide')

if 'jwt' not in st.session_state:
    st.session_state.jwt, st.session_state.username, st.session_state.role = None, None, None

with get_db_session() as db:
    if db.query(User).count() == 0:
        u = User(username='admin', password_hash=hash_password('admin123'), role='admin')
        db.add(u)
        st.info('Default admin user created. Login with admin / admin123')

if not st.session_state.jwt:
    st.title('🔐 Equity Tracker Login')
    with st.form("login_form"):
        login_user, login_pw = st.text_input('Username'), st.text_input('Password', type='password')
        if st.form_submit_button('Login', use_container_width=True):
            with get_db_session() as db:
                user = db.query(User).filter(User.username == login_user).first()
                if user and verify_password(login_pw, user.password_hash):
                    st.session_state.jwt = create_jwt(user.username, user.role)
                    st.session_state.username, st.session_state.role = user.username, user.role
                    st.rerun()
                else: st.error('❌ Invalid credentials')
    st.stop()

payload = decode_jwt(st.session_state.jwt)
if not payload:
    st.error('⏰ Session invalid or expired. Please re-login.'), st.session_state.clear()
    if st.button('Back to Login'): st.rerun()
    st.stop()

current_user, current_role = st.session_state.username, st.session_state.role

header_cols = st.columns([3, 1])
with header_cols[0]: st.title('📊 Equity Tracker')
with header_cols[1]:
    with st.container(border=True):
        st.markdown(f"👤 **{current_user}** ({current_role})")
        if st.button('🚪 Logout', use_container_width=True): st.session_state.clear(); st.rerun()
st.divider()

user_investor_name = None
with get_db_session() as db:
    user_obj = db.query(User).filter(User.username == current_user).first()
    if user_obj: user_investor_name = user_obj.investor_name

# Fetch all events to calculate totals, regardless of role
with get_db_session() as db:
    all_events_df = pd.read_sql(db.query(Event).statement, db.bind)
    all_events_df['date'] = pd.to_datetime(all_events_df['date']).dt.date

if all_events_df.empty:
    st.info('No events in the system yet. The admin needs to add the first event.')
    st.stop()

# --- LOGICA DI VISUALIZZAZIONE BASATA SUL RUOLO ---

# Calcola sempre i totali reali basandosi su tutti gli eventi
total_balances, total_history = replay_events(all_events_df)
total_portfolio_value = sum(total_balances.values())

# Filtra gli eventi per la vista dell'utente, se non è admin
if current_role != 'admin':
    if user_investor_name:
        events_df = all_events_df[
            (all_events_df['type'] == 'valuation') | (all_events_df['investor'] == user_investor_name)
        ]
    else: # Un utente senza nome investitore associato vede solo le valutazioni
        events_df = all_events_df[all_events_df['type'] == 'valuation']
else:
    events_df = all_events_df # L'admin vede tutto

# --- INIZIO LAYOUT ---

# Calcola le metriche
overall_roi = 0
if total_portfolio_value > 0:
    total_usd_invested_overall = all_events_df[all_events_df['type'] == 'deposit']['usd_amount'].sum()
    if total_usd_invested_overall > 0:
        overall_roi = ((total_portfolio_value - total_usd_invested_overall) / total_usd_invested_overall * 100)

metric_cols = st.columns(3)
metric_cols[0].metric('💵 Current Portfolio Total (USD)', f'${total_portfolio_value:,.2f}')
metric_cols[1].metric('📈 Overall ROI', f'{overall_roi:,.2f}%')

# Metrica personalizzata per l'utente
if current_role != 'admin' and user_investor_name:
    user_balance = total_balances.get(user_investor_name, 0.0)
    user_share = (user_balance / total_portfolio_value * 100) if total_portfolio_value > 0 else 0
    metric_cols[2].metric('✨ La Tua Quota', f'{user_share:,.2f}%')
else:
    visible_investors = sorted({inv for inv in events_df['investor'].dropna().unique()})
    metric_cols[2].metric('👥 Investors', len(visible_investors))

# Definisci le tab
tab_list = ["📈 Dashboard", "👥 Dettaglio Investitori"]
if current_role == 'admin':
    tab_list.append("⚙️ Gestione Eventi")

tabs = st.tabs(tab_list)

# --- Tab 1: Dashboard ---
with tabs[0]:
    st.subheader("Andamento del Valore Totale del Portafoglio (USD)")
    # Questo grafico è lo stesso per tutti e si basa sulla storia totale
    if not total_history.empty:
        fig_portfolio = px.line(total_history, x='date', y='total', markers=True, labels={'date': 'Data', 'total': 'Valore Totale (USD)'})
        st.plotly_chart(fig_portfolio, use_container_width=True)

    # Grafici condizionali basati sul ruolo
    if current_role == 'admin':
        st.subheader("Valore delle Quote Individuali (USD) nel Tempo")
        if not total_history.empty and len(total_history.columns) > 2:
            investor_cols = [c for c in total_history.columns if c not in ['date', 'total']]
            if investor_cols:
                value_df_melted = total_history.melt(id_vars='date', value_vars=investor_cols, var_name='Investor', value_name='USD Value')
                fig_investor_value = px.line(value_df_melted, x='date', y='USD Value', color='Investor', markers=True, labels={'USD Value': 'Valore Quota (USD)', 'date': 'Data'})
                st.plotly_chart(fig_investor_value, use_container_width=True)
    else: # Vista utente
        st.subheader("Andamento del Tuo Valore (USD)")
        if user_investor_name and user_investor_name in total_history.columns:
            user_history_df = total_history[['date', user_investor_name]].copy()
            user_history_df.rename(columns={user_investor_name: 'Il Tuo Valore (USD)'}, inplace=True)
            fig_user_value = px.line(user_history_df, x='date', y='Il Tuo Valore (USD)', markers=True)
            st.plotly_chart(fig_user_value, use_container_width=True)

    st.subheader("Evoluzione Quote Investitori (%)")
    # L'admin vede tutti, l'utente vede solo se stesso
    history_to_show = total_history if current_role == 'admin' else total_history[['date', 'total', user_investor_name]] if user_investor_name in total_history.columns else pd.DataFrame()
    if not history_to_show.empty and len(history_to_show.columns) > 2:
        inv_cols = [c for c in history_to_show.columns if c not in ['date', 'total']]
        if inv_cols:
            pct_df = history_to_show[inv_cols].div(history_to_show['total'], axis=0).fillna(0)
            pct_df['date'] = history_to_show['date']
            pct_df_melted = pct_df.melt(id_vars='date', var_name='Investor', value_name='Share')
            fig_shares = px.area(pct_df_melted, x='date', y='Share', color='Investor', markers=True)
            fig_shares.update_yaxes(tickformat='.0%')
            st.plotly_chart(fig_shares, use_container_width=True)

# --- Tab 2: Dettaglio Investitori ---
with tabs[1]:
    st.subheader("Riepilogo Investitori")
    
    # L'admin vede tutti, l'utente vede solo la sua riga
    investors_to_show = sorted({inv for inv in events_df['investor'].dropna().unique()})
    
    investor_data = []
    for inv in investors_to_show:
        inv_deposits = all_events_df[(all_events_df['investor'] == inv) & (all_events_df['type'] == 'deposit')]
        inv_withdrawals = all_events_df[(all_events_df['investor'] == inv) & (all_events_df['type'] == 'withdrawal')]
        total_usd_invested = inv_deposits['usd_amount'].sum()
        total_withdrawn_usd = inv_withdrawals['usd_amount'].sum()
        # Usa i bilanci totali per i calcoli corretti
        current_usd_value = total_balances.get(inv, 0.0)
        roi_pct = ((current_usd_value + total_withdrawn_usd - total_usd_invested) / total_usd_invested * 100) if total_usd_invested > 0 else 0
        
        investor_data.append({
            'Investor': inv, 'EUR Invested': inv_deposits['eur_amount'].sum(), 'USD Invested': total_usd_invested,
            'Total Withdrawn (USD)': total_withdrawn_usd, 'Current Value (USD)': current_usd_value,
            'Share %': (current_usd_value / total_portfolio_value * 100) if total_portfolio_value > 0 else 0, 'ROI %': roi_pct
        })
    
    if investor_data:
        df_inv = pd.DataFrame(investor_data)
        def roi_color(val):
            style = 'color: black;'
            if val > 0:
                norm_val = val / max_roi if max_roi > 0 else 0; red, green = int(255 * (1 - norm_val)), 255
                return f'background-color: rgb({red}, {green}, 0); {style}'
            elif val < 0:
                norm_val = val / min_roi if min_roi < 0 else 0; red, green = 255, int(255 * (1 - norm_val))
                return f'background-color: rgb({red}, {green}, 0); {style}'
            else: return f'background-color: yellow; {style}'
        max_roi, min_roi = df_inv['ROI %'].max(), df_inv['ROI %'].min()
        st.dataframe(df_inv.style.format({
            'EUR Invested': '€{:,.2f}', 'USD Invested': '${:,.2f}', 'Total Withdrawn (USD)': '${:,.2f}',
            'Current Value (USD)': '${:,.2f}', 'Share %': '{:.2f}%', 'ROI %': '{:+.2f}%'
        }).apply(lambda x: x.map(roi_color), subset=['ROI %']), use_container_width=True, hide_index=True)

# --- Tab 3: Gestione Eventi (Solo Admin) ---
if current_role == 'admin':
    with tabs[2]:
        form_cols = st.columns(2)
        with form_cols[0]:
            st.subheader("➕ Aggiungi Evento")
            dep_tab, wd_tab, val_tab = st.tabs(["💰 Deposito", "💸 Prelievo", "📈 Valutazione"])
            # ... (Codice dei form rimane identico) ...
            with dep_tab:
                with st.form("deposit_form"):
                    d_date = st.date_input('Data', datetime.date.today(), key='d_date')
                    investor, eur_amount = st.text_input('Nome Investitore', value=user_investor_name or "", key='d_inv'), st.number_input('Importo (EUR)', min_value=0.01, step=100.0, key='d_eur')
                    if st.form_submit_button('Salva Deposito', use_container_width=True):
                        rate, usd_amount = get_historical_eurusd(d_date), eur_amount * get_historical_eurusd(d_date)
                        with get_db_session() as db: db.add(Event(date=d_date, type='deposit', investor=investor.strip(), eur_amount=eur_amount, usd_amount=usd_amount, eurusd_rate=rate))
                        st.success(f'Deposito salvato.'); time.sleep(1); st.rerun()
            with wd_tab:
                with st.form("withdrawal_form"):
                    w_date, w_investor = st.date_input('Data', datetime.date.today(), key='w_date'), st.text_input('Nome Investitore', value=user_investor_name or "", key='w_inv')
                    w_usd_amount = st.number_input('Importo (USD)', min_value=0.01, step=100.0, key='w_usd')
                    if st.form_submit_button('Salva Prelievo', use_container_width=True):
                        rate, eur_amount = get_historical_eurusd(w_date), w_usd_amount / get_historical_eurusd(w_date)
                        with get_db_session() as db: db.add(Event(date=w_date, type='withdrawal', investor=w_investor.strip(), eur_amount=eur_amount, usd_amount=w_usd_amount, eurusd_rate=rate))
                        st.success(f'Prelievo salvato.'); time.sleep(1); st.rerun()
            with val_tab:
                with st.form("valuation_form"):
                    v_date, v_total = st.date_input('Data', datetime.date.today(), key='v_date'), st.number_input('Valore Totale Portafoglio (USD)', min_value=0.01, step=1000.0, key='v_usd')
                    if st.form_submit_button('Salva Valutazione', use_container_width=True):
                        with get_db_session() as db: db.add(Event(date=v_date, type='valuation', valuation_total_usd=v_total))
                        st.success(f'Valutazione salvata.'); time.sleep(1); st.rerun()
        with form_cols[1]:
            st.subheader("✏️ Modifica / Elimina Eventi")
            with get_db_session() as db:
                # L'admin vede tutti gli eventi per la modifica
                events_to_edit = pd.read_sql(db.query(Event).statement, db.bind)
                if not events_to_edit.empty:
                    st.dataframe(events_to_edit.sort_values('date', ascending=False), height=250, use_container_width=True, hide_index=True)
                    ev_id_to_edit = st.selectbox('Seleziona ID Evento', options=events_to_edit['id'].tolist())
                    if ev_id_to_edit:
                        ev_row = db.get(Event, ev_id_to_edit)
                        if ev_row:
                            with st.form(f"edit_form_{ev_row.id}"):
                                st.markdown(f"**Modifica Evento #{ev_row.id} ({ev_row.type})**")
                                e_date = st.date_input('Data', value=pd.to_datetime(ev_row.date))
                                if ev_row.type == 'deposit':
                                    e_investor, e_eur = st.text_input('Nome Investitore', value=ev_row.investor or ""), st.number_input('Importo EUR', value=ev_row.eur_amount or 0.0)
                                elif ev_row.type == 'withdrawal':
                                    e_investor, e_usd = st.text_input('Nome Investitore', value=ev_row.investor or ""), st.number_input('Importo USD', value=ev_row.usd_amount or 0.0)
                                else: e_val = st.number_input('Valore Totale USD', value=ev_row.valuation_total_usd or 0.0)
                                c1, c2 = st.columns(2)
                                if c1.form_submit_button('💾 Aggiorna', use_container_width=True):
                                    ev_row.date, rate = e_date, get_historical_eurusd(e_date)
                                    if ev_row.type in ['deposit', 'withdrawal']: ev_row.investor = e_investor.strip() or None
                                    if ev_row.type == 'deposit': ev_row.eurusd_rate, ev_row.usd_amount, ev_row.eur_amount = rate, e_eur * rate, e_eur
                                    elif ev_row.type == 'withdrawal': ev_row.eurusd_rate, ev_row.eur_amount, ev_row.usd_amount = rate, e_usd / rate, e_usd
                                    else: ev_row.valuation_total_usd = e_val
                                    db.commit(); st.success(f"Evento #{ev_row.id} aggiornato."); time.sleep(1); st.rerun()
                                if c2.form_submit_button('🗑️ Elimina', type="primary", use_container_width=True):
                                    db.delete(ev_row); db.commit(); st.success(f"Evento #{ev_row.id} eliminato."); time.sleep(1); st.rerun()

# Pannello Admin (sempre in fondo)
if current_role == 'admin':
    with st.expander('👑 Pannello Admin: Gestione Utenti'):
        # ... (Codice del Pannello Admin rimane identico) ...
        admin_cols = st.columns(2)
        with admin_cols[0]:
            st.subheader('➕ Crea Nuovo Utente')
            with st.form("create_user_form"):
                new_username, new_password = st.text_input('Username'), st.text_input('Password', type='password')
                new_role, new_investor_name = st.selectbox('Ruolo', ['user', 'admin']), st.text_input('Nome Investitore (opzionale)')
                if st.form_submit_button('Crea Utente', use_container_width=True):
                    if new_username and new_password:
                        user_created = False
                        try:
                            with get_db_session() as db:
                                if db.query(User).filter(User.username == new_username).first(): st.error('Username già esistente.')
                                else:
                                    db.add(User(username=new_username.strip(), password_hash=hash_password(new_password), role=new_role, investor_name=new_investor_name.strip() or None))
                                    user_created = True
                        except Exception as e: st.error(f"Errore database: {e}")
                        if user_created: st.success(f'Utente "{new_username}" creato.'); time.sleep(1); st.rerun()
                    else: st.warning('Username e password sono richiesti.')
        with admin_cols[1]:
            st.subheader('✏️ Modifica o Elimina Utente')
            with get_db_session() as db:
                all_users = db.query(User).all()
                user_map = {f"{user.username} (ID: {user.id})": user.id for user in all_users}
                if not user_map: st.info("Nessun utente da modificare.")
                else:
                    selected_user_key = st.selectbox("Seleziona utente", options=user_map.keys())
                    if selected_user_key:
                        user_to_edit = db.get(User, user_map[selected_user_key])
                        with st.form("edit_user_form"):
                            is_admin_user = (user_to_edit.username == 'admin')
                            st.markdown(f"**Gestione Utente: {user_to_edit.username}**")
                            edited_username = st.text_input("Username", value=user_to_edit.username, disabled=is_admin_user)
                            edited_role = st.selectbox("Ruolo", ['user', 'admin'], index=1 if user_to_edit.role == 'admin' else 0, disabled=is_admin_user)
                            edited_inv_name = st.text_input("Nome Investitore", value=user_to_edit.investor_name or "", disabled=is_admin_user)
                            new_password = st.text_input("Nuova Password (lascia vuoto per non cambiare)", type="password")
                            c1, c2 = st.columns(2)
                            if c1.form_submit_button("💾 Aggiorna Utente", use_container_width=True):
                                if not is_admin_user:
                                    user_to_edit.investor_name = edited_inv_name.strip() or None
                                    if edited_username != user_to_edit.username and db.query(User).filter(User.username == edited_username).first(): st.error("Username già in uso.")
                                    else: user_to_edit.username, user_to_edit.role = edited_username, edited_role
                                if new_password: user_to_edit.password_hash = hash_password(new_password)
                                db.commit(); st.success(f"Utente '{user_to_edit.username}' aggiornato."); time.sleep(1); st.rerun()
                            if not is_admin_user and c2.form_submit_button("🗑️ ELIMINA UTENTE", type="primary", use_container_width=True):
                                db.delete(user_to_edit); db.commit(); st.warning(f"Utente '{user_to_edit.username}' eliminato."); time.sleep(1); st.rerun()
        st.divider()
        st.subheader('👥 Utenti Attuali')
        with get_db_session() as db:
            st.dataframe(pd.read_sql(db.query(User).statement, db.bind)[['id', 'username', 'role', 'investor_name']], use_container_width=True, hide_index=True)