#%%
import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.ensemble import RandomForestRegressor
import psycopg2
from sshtunnel import SSHTunnelForwarder
import os
from dotenv import load_dotenv
import joblib  # for saving/loading model

#%%
# ---------- Load environment variables ----------
load_dotenv()

BATCH_SIZE = 5000
LOOKBACK_DAYS = 30
RAISED_LOOKBACK = 15
MODEL_PATH_CONV = "./models/rf_model_conversions.pkl"
MODEL_PATH_REV = "./models/rf_model_revenue.pkl"

#%%

# ---------- CACHED SSH TUNNEL ----------
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
    """Fetch data from remote PostgreSQL DB via SSH tunnel in batches."""
    tunnel = get_ssh_tunnel()
    conn = get_db_connection(tunnel)
    base_query = """
    SELECT 
        tickets.id as ticket_id,
        tickets.created_at as ticket_created_at,
        quotations.id as quotations_id,
        quotations.active as quotations_active,
        invoices.id as invoices_id,
        (invoices.professional_fees - invoices.total_discount_on_professional_fees) as invoice_revenue,
        quotations.professional_fees as quotations_professional_fees,
        quotations.total_discount_on_professional_fees as quotations_total_discount_on_professional_fees,
        quotations.total_quotation_amount as quotations_total_quotation_amount,
        invoices.invoice_total_amount as invoices_invoice_total_amount,
        invoices.created_at as payment_date,
        quotations.created_at as quotation_raised_at
    FROM quotations
    LEFT JOIN tickets ON quotations.ticket_id = tickets.id
    LEFT JOIN invoices ON invoices.quotation_id = quotations.id
    WHERE DATE(quotations.created_at) >= '2024-09-01'
    """

    all_chunks = []
    last_id = 0

    while True:
        paged_query = f"""
        {base_query}
        AND tickets.id > {last_id}
        ORDER BY tickets.id
        LIMIT {BATCH_SIZE};
        """
        chunk = pd.read_sql(paged_query, conn)
        if chunk.empty:
            break
        all_chunks.append(chunk)
        last_id = chunk['ticket_id'].max()

    df = pd.concat(all_chunks, ignore_index=True)
    conn.close()
    return df

#%%
def prepare_features(df):
    """Prepare daily aggregation + recently raised stats + weekday features."""
    today = pd.Timestamp.now().normalize().date()
    df['raised_date'] = df['quotation_raised_at'].dt.date

    # Daily closed quotations
    closed = df[df['payment_date'].notna()].copy()
    closed['conversion_date'] = closed['payment_date'].dt.date
    daily = closed.groupby('conversion_date').agg(
        conversions=('payment_date', 'count'),
        revenue=('invoices_invoice_total_amount', 'sum')
    ).reset_index().sort_values('conversion_date')

    # Recently raised stats
    recently_raised_counts = []
    recently_converted_counts = []
    for conv_date in daily['conversion_date']:
        start_date = pd.to_datetime(conv_date) - pd.Timedelta(days=RAISED_LOOKBACK)
        end_date = pd.to_datetime(conv_date)
        yesterday = end_date - pd.Timedelta(days=1)

        # recently raised and still active yesterday
        count_recently_raised = df[
            (df['raised_date'] >= start_date.date()) &
            (df['raised_date'] < end_date.date()) &
            ((df['payment_date'].isna()) | (df['payment_date'].dt.date > yesterday.date()))
        ].shape[0]
        recently_raised_counts.append(count_recently_raised)

        # recently raised and converted today
        count_recently_converted = closed[
            (closed['conversion_date'] == end_date.date()) &
            (closed['raised_date'] >= start_date.date()) &
            (closed['raised_date'] < end_date.date())
        ].shape[0]
        recently_converted_counts.append(count_recently_converted)

    daily['recently_raised_quotations'] = recently_raised_counts
    daily['recently_raised_then_converted'] = recently_converted_counts

    # Weekday number
    daily['week_day_num'] = pd.to_datetime(daily['conversion_date']).dt.dayofweek + 1

    today_data = daily[daily["conversion_date"] == today].copy()
    daily = daily[daily["conversion_date"] < today]  # exclude from training


    # Keep last LOOKBACK_DAYS
    recent = daily.tail(LOOKBACK_DAYS).copy()
    recent['day_num'] = (pd.to_datetime(recent['conversion_date']) - pd.to_datetime(recent['conversion_date'].min())).dt.days

    # Features
    X = recent[['day_num', 'week_day_num', 'recently_raised_quotations', 'recently_raised_then_converted']].fillna(0)
    y_conv = recent['conversions']
    y_rev = recent['revenue']
    return X, y_conv, y_rev, recent

#%%
def train_and_save_models():
    """Fetch data, prepare features, train RandomForest, and save models."""
    df = fetch_data_from_db()
    X, y_conv, y_rev, recent = prepare_features(df)

    model_conv = RandomForestRegressor(n_estimators=200, random_state=42)
    model_rev = RandomForestRegressor(n_estimators=200, random_state=42)

    model_conv.fit(X, y_conv)
    model_rev.fit(X, y_rev)

    os.makedirs('./models', exist_ok=True)
    joblib.dump(model_conv, MODEL_PATH_CONV)
    joblib.dump(model_rev, MODEL_PATH_REV)
    return model_conv, model_rev, recent

#%%
# ---------- Streamlit App ----------
st.title("📊 Revenue & Conversion Forecasting")

st.markdown("""
This app predicts today's expected revenue and expected conversion counts based on historical quotations activity and recently raised active quotations.
""")

if st.button("🔁 Retrain model with latest data"):
    with st.spinner("Fetching data and training models..."):
        model_conv, model_rev, recent = train_and_save_models()
        st.session_state['recent'] = recent
        st.session_state['model_conv'] = model_conv
        st.session_state['model_rev'] = model_rev
    st.success("✅ Model retrained and saved successfully!")
# Display recent data table
#%%
st.subheader("📅 Predict for Today")

if st.button("➡️ Predict Today's Conversions & Revenue"):
    if 'recent' in st.session_state and os.path.exists(MODEL_PATH_CONV) and os.path.exists(MODEL_PATH_REV):
        recent = st.session_state['recent']
        # Load trained models if not in memory
        model_conv = joblib.load(MODEL_PATH_CONV)
        model_rev = joblib.load(MODEL_PATH_REV)

        today = pd.Timestamp.now().normalize().date()
        today_num = (today - pd.to_datetime(recent["conversion_date"].min()).date()).days
        today_week_num = pd.Timestamp(today).dayofweek + 1

        latest_behavior = recent[[
            "recently_raised_quotations",
            "recently_raised_then_converted"
        ]].iloc[-1].values.reshape(1, -1)

        X_today = np.hstack([[today_num, today_week_num], latest_behavior.flatten()]).reshape(1, -1)

        pred_conversions = model_conv.predict(X_today)[0]
        pred_revenue = model_rev.predict(X_today)[0]

        pred_conversions = max(pred_conversions, 0)
        pred_revenue = max(pred_revenue, 0)

        # Display in Streamlit
        st.success(f"Expected Conversions: {pred_conversions:.0f}")
        st.success(f"Expected Revenue: ₹{pred_revenue:,.2f}")

        # Optional: show recent training data
        st.subheader("Recent Training Data (last 7 days)")
        st.dataframe(
            recent[[
                "conversion_date", "conversions", "revenue",
                "recently_raised_quotations", "recently_raised_then_converted",
                "week_day_num"
            ]].tail(7)
        )
    else:
        st.warning("No trained model available. Please retrain first!")

