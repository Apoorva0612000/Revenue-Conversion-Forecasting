#%%
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib
import psycopg2
from sshtunnel import SSHTunnelForwarder
import os
from dotenv import load_dotenv
from datetime import datetime

#%%
st.title("📊 Quotation Conversion Days Prediction")

st.markdown("""
This app predicts the number of days it will take for a quotation to convert into a paid invoice.
""")

#%% ---------- CONFIG ----------
MODEL_PATH = "./models/gb_conversion_model.pkl"
SCALER_PATH = "./models/scaler.pkl"

load_dotenv()

#%%
# ---------- DB FETCH FUNCTION ----------

# ---------- DB CONNECTION (CACHED) ----------
@st.cache_resource
def get_ssh_tunnel():
    """Start SSH tunnel once per Streamlit session."""
    tunnel = SSHTunnelForwarder(
        (os.getenv('SSH_HOST'), int(os.getenv('SSH_PORT'))),
        ssh_username=os.getenv('SSH_USER'),
        ssh_password=os.getenv('SSH_PASSWORD'),
        remote_bind_address=(os.getenv('JUMPBOX_IP'), int(os.getenv('PORT'))),
        local_bind_address=(os.getenv('DB_HOST'), int(os.getenv('DB_PORT')))
    )
    tunnel.start()
    return tunnel


# ---------- FRESH DB CONNECTION ----------
def get_db_connection(tunnel):
    """Open a new PostgreSQL connection via the cached tunnel."""
    conn = psycopg2.connect(
        dbname=os.getenv('DB_NAME'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
        host=os.getenv('DB_HOST'),  # connect via local tunnel
        port=os.getenv('DB_PORT')
    )
    return conn
def fetch_data_from_db():
    """Fetch calls, employee quotations, and ticket quotations from database"""
    tunnel = get_ssh_tunnel()
    conn = get_db_connection(tunnel)
        # Queries
    ticket_q = pd.read_sql("""
                SELECT 
                tickets.id as ticket_id,
                quotations.id as quotations_id,
                (quotations.professional_fees - quotations.total_discount_on_professional_fees) as invoice_revenue,
                quotations.professional_fees as quotations_professional_fees,
                quotations.total_discount_on_professional_fees as quotations_total_discount_on_professional_fees,
                quotations.total_quotation_amount as quotations_total_quotation_amount,
                invoices.created_at as payment_date,
                quotations.created_at as quotation_raised_at
            FROM quotations
            LEFT JOIN tickets ON quotations.ticket_id = tickets.id
            LEFT JOIN invoices ON invoices.quotation_id = quotations.id
            WHERE DATE(quotations.created_at) >= '2025-01-01'
    """, conn)
    
    calls = pd.read_sql("""
        SELECT 
        tickets.id as ticket_id,
        outbound_tickets.employee_id as caller_id,
        outbound_tickets.talktime
    FROM tickets
    LEFT JOIN outbound_tickets ON outbound_tickets.ticket_id = tickets.id
    WHERE tickets.created_at >= '2024-09-01'
    """, conn)
    
    conn.close()
    return ticket_q, calls

#%%
# ---------- TRAINING FUNCTION ----------
def train_model(ticket_q, calls):
    # Preprocess ticket data
    ticket_q["quotation_raised_at"] = pd.to_datetime(ticket_q["quotation_raised_at"])
    ticket_q["payment_date"] = pd.to_datetime(ticket_q["payment_date"])
    ticket_q["conversion_days"] = (ticket_q["payment_date"] - ticket_q["quotation_raised_at"]).dt.days
    ticket_q = ticket_q.dropna(subset=["conversion_days"])

    ticket_q["Discount_Percentage"] = (
        ticket_q["quotations_total_discount_on_professional_fees"] /
        ticket_q["quotations_professional_fees"]
    ).fillna(0)
    ticket_q["Day_Of_Week"] = ticket_q["quotation_raised_at"].dt.dayofweek

    # Aggregate calls
    call_features = calls.groupby(["ticket_id", "caller_id"]).agg(
        num_calls=("ticket_id", "count"),
        avg_talktime=("talktime", "mean")
    ).reset_index()

    call_features["avg_talktime"] = call_features["avg_talktime"] / 60

    df = ticket_q.merge(call_features, on="ticket_id", how="left")
    df["num_calls"] = df["num_calls"].fillna(0)
    df["avg_talktime"] = df["avg_talktime"].fillna(0)

    # Features and target
    X = df[[
        "quotations_total_quotation_amount",
        "Discount_Percentage",
        "Day_Of_Week",
        "num_calls",
        "avg_talktime"
    ]]
    y = df["conversion_days"]
    quotation_ids = df["quotations_id"]

    np.isfinite(X).all()
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.isna().sum()
    X = X.fillna(0)

    # Scaling
    ss = StandardScaler()
    X_scaled = ss.fit_transform(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Model
    model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)

    st.success(f"✅ Model trained successfully! MAE on test set: {mae:.2f} days")

    # Save model and scaler
    os.makedirs('./models', exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(ss, SCALER_PATH)

    # Store in session state
    st.session_state['model'] = model
    st.session_state['scaler'] = ss
    st.session_state['model_mae'] = mae

#%%
# ---------- RETRAIN BUTTON ----------
if st.button("🔁 Retrain Model with Latest Data"):
    with st.spinner("Fetching data and training model..."):
        ticket_q, calls= fetch_data_from_db()
        train_model(ticket_q, calls)

#%%
# ---------- LOAD MODEL IF EXISTS ----------
if 'model' not in st.session_state:
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        st.session_state['model'] = joblib.load(MODEL_PATH)
        st.session_state['scaler'] = joblib.load(SCALER_PATH)
    else:
        st.warning("No trained model available. Please retrain first!")

#%%
#%%
# ---------- PREDICT USING QUOTATION ID ----------
st.subheader("🔍 Predict Conversion Days by Quotation ID")

quotation_id_input = st.text_input("Enter Quotation ID to Predict")

def fetch_quotation_details(quotation_id):
    """Fetch quotation and call info for a specific quotation ID"""
    tunnel = get_ssh_tunnel()
    conn = get_db_connection(tunnel)
        # Get quotation details
    q = pd.read_sql(f"""
        SELECT 
            q.id AS quotations_id,
            q.ticket_id as ticket_id,
            (q.professional_fees - q.total_discount_on_professional_fees) as invoice_revenue,
            q.professional_fees as quotations_professional_fees,
            q.total_discount_on_professional_fees as quotations_total_discount_on_professional_fees,
            q.total_quotation_amount as quotations_total_quotation_amount,
            i.created_at as payment_date,
            q.created_at as quotation_raised_at
        FROM quotations q
        LEFT JOIN invoices i ON i.quotation_id = q.id
        WHERE q.id = '{quotation_id}'
    """, conn)

    # Get related call data
    calls = pd.read_sql(f"""
        SELECT 
        t.id as ticket_id,
        o.employee_id as caller_id,
        o.talktime,
        o.created_at as Dialed_at                
        FROM tickets t
        LEFT JOIN outbound_tickets o ON o.ticket_id = t.id
        WHERE t.id IN (SELECT ticket_id FROM quotations WHERE id = '{quotation_id}')
    """, conn)

    conn.close()

    if q.empty:
        return None, None

    # Merge quotation + call features
    q = q.merge(calls, on="ticket_id", how="left")
    q['num_calls'] = q.groupby('quotations_id')['dialed_at'].transform('count')
    q['avg_talktime'] = q.groupby('quotations_id')['talktime'].transform('mean')

    q["avg_talktime"] = q["avg_talktime"] / 60

    # q["num_calls"] = q["num_calls"].fillna(0)
    # q["avg_talktime"] = q["avg_talktime"].fillna(0)

    # Compute discount %
    q["Discount_Percentage"] = (
        q["quotations_total_discount_on_professional_fees"] / q["quotations_professional_fees"]
    ).fillna(0)
    q["Day_Of_Week"] = pd.to_datetime(q["quotation_raised_at"]).dt.dayofweek

    return q, calls

#%%
# ---------- PREDICT BUTTON ----------
if st.button("🔎 Fetch and Predict"):
    if not quotation_id_input.strip():
        st.warning("Please enter a valid Quotation ID.")
    else:
        q, _ = fetch_quotation_details(quotation_id_input)
        if q is None:
            st.error("Quotation ID not found.")
        else:
            st.dataframe(q[[
                "quotations_id", "quotation_raised_at", "quotations_professional_fees",
                "Discount_Percentage", "Day_Of_Week", "num_calls", "avg_talktime", "payment_date"
            ]])

            # Check quotation status
            quotation_date = pd.to_datetime(q.iloc[0]["quotation_raised_at"])
            payment_date = q.iloc[0]["payment_date"]

            if pd.notnull(payment_date):
                st.info("💰 This quotation is already converted (inactive).")
            else:
                days_active = (datetime.now() - quotation_date).days
                if days_active > 30:
                    st.warning(f"⚠️ This quotation has been active for over {days_active} days without payment.")
                else:
                    if 'model' in st.session_state and 'scaler' in st.session_state:
                        q['num_calls'] = q['num_calls'].fillna(0)
                        q['avg_talktime'] = q['avg_talktime'].fillna(0)
                        X_new = q[[
                            "quotations_total_quotation_amount",
                            "Discount_Percentage",
                            "Day_Of_Week",
                            "num_calls",
                            "avg_talktime"
                        ]]
                        X_scaled_new = st.session_state['scaler'].transform(X_new)
                        pred_days = st.session_state['model'].predict(X_scaled_new)[0]

                        # Retrieve or estimate MAE
                        mae = st.session_state.get("model_mae", 3.5)  # default fallback if not stored

                        st.success(f"⏳ Predicted Conversion Time: {pred_days:.1f} ± {mae:.1f} days")
                        st.caption("ℹ️ ± represents average model error (Mean Absolute Error).")
                    else:
                        st.warning("Model not available. Please retrain first.")
