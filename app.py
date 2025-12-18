import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Lazy import for SHAP to avoid import errors at startup
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError as e:
    SHAP_AVAILABLE = False
    st.warning(f"SHAP not available: {e}. Explanations will be limited.")

# Page configuration
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Remove default padding from top and move header to absolute top
# Force light mode and disable dark mode
st.markdown("""
    <style>
    /* Force light mode - disable dark mode */
    .stApp {
        color-scheme: light;
    }
    [data-testid="stAppViewContainer"] {
        color-scheme: light;
    }
    /* Hide theme settings */
    [data-testid="stHeader"] [data-testid="stDecoration"] {
        display: none;
    }
    /* Force light background */
    .stApp {
        background: linear-gradient(135deg, #F5E6D3 0%, #E8D5B7 100%) !important;
    }
    
    .stApp > header {
        visibility: hidden;
        height: 0;
        padding: 0;
        margin: 0;
    }
    .stApp {
        padding-top: 0 !important;
        margin-top: 0 !important;
    }
    .main .block-container {
        padding-top: 0.1rem !important;
        padding-bottom: 2rem;
    }
    #MainMenu {
        visibility: hidden;
        height: 0;
    }
    </style>
""", unsafe_allow_html=True)

# Custom CSS with brown color scheme
st.markdown("""
    <style>
    /* Main background - cream color */
    .stApp {
        background: linear-gradient(135deg, #F5E6D3 0%, #E8D5B7 100%);
    }
    
    /* Sidebar background - dark brown */
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #5D4037 0%, #3E2723 100%);
    }
    
    [data-testid="stSidebar"] > div:first-child {
        background: linear-gradient(135deg, #5D4037 0%, #3E2723 100%);
    }
    
    /* Sidebar text color - white/cream for visibility */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p {
        color: #FFF8E1 !important;
    }
    
    /* Sidebar selectbox and input styling */
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stNumberInput label,
    [data-testid="stSidebar"] .stSlider label {
        color: #FFF8E1 !important;
    }
    
    /* Header card with bold brown background - extra large size */
    .header-card {
        background: linear-gradient(135deg, #5D4037 0%, #3E2723 100%);
        padding: 4rem 3rem;
        border-radius: 20px;
        border: 4px solid #8D6E63;
        margin-bottom: 3rem;
        box-shadow: 0 8px 16px rgba(0,0,0,0.4);
        position: relative;
        width: 100%;
    }
    
    /* Main header */
    .main-header {
        font-size: 3.5rem;
        font-weight: bold;
        color: #FFF8E1;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    /* Subtitle */
    .subtitle {
        font-size: 1.2rem;
        color: #FFF8E1;
        text-align: center;
        margin-bottom: 2.5rem;
        font-style: italic;
        padding: 0 1rem;
    }
    
    /* Feature cards - dark brown background with white text */
    .feature-card {
        background: linear-gradient(135deg, #5D4037 0%, #3E2723 100%);
        padding: 0.6rem 0.5rem;
        border-radius: 8px;
        border: 1px solid #8D6E63;
        text-align: center;
        color: #FFF8E1;
        font-weight: 500;
        font-size: 1.1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: transform 0.2s;
        min-height: 40px;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .feature-card:hover {
        transform: translateY(-1px);
        box-shadow: 0 3px 6px rgba(0,0,0,0.15);
    }
    
    /* Input boxes */
    .input-box {
        background: linear-gradient(135deg, #EFEBE9 0%, #D7CCC8 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #A1887F;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    
    .input-box h3 {
        color: #5D4037;
        margin-bottom: 1rem;
        border-bottom: 2px solid #8D6E63;
        padding-bottom: 0.5rem;
    }
    
    /* Column borders */
    .column-border {
        border-right: 2px solid #8D6E63;
        padding-right: 1.5rem;
        margin-right: 0.5rem;
    }
    
    /* Reduced font size for form elements - match Amount & Fee size */
    .small-form {
        font-size: 0.85rem;
    }
    
    .small-form label {
        font-size: 0.8rem !important;
    }
    
    .small-form .stSelectbox label,
    .small-form .stNumberInput label,
    .small-form .stSlider label {
        font-size: 0.8rem !important;
    }
    
    /* Make all form elements more compact */
    .small-form .stSelectbox,
    .small-form .stNumberInput,
    .small-form .stSlider {
        font-size: 0.85rem;
    }
    
    /* Compact section headings */
    .small-form h3 {
        font-size: 1.1rem;
        margin-bottom: 0.8rem;
    }
    
    /* Style for risk indicators - stacked rows */
    .risk-indicator-row {
        margin-bottom: 1rem;
    }
    
    /* Compact slider styling - match Amount & Fee size */
    .compact-slider {
        font-size: 0.85rem;
    }
    
    .compact-slider .stSlider {
        font-size: 0.8rem;
    }
    
    .compact-slider label {
        font-size: 0.8rem !important;
    }
    
    .compact-slider h3 {
        font-size: 1.1rem;
        margin-bottom: 0.8rem;
    }
    
    /* Transaction details two-column layout */
    .txn-details-col {
        padding-right: 1rem;
    }
    
    /* Two-column layout for transaction details using CSS */
    .txn-details-container {
        width: 100%;
    }
    
    .txn-details-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0 1rem;
        align-items: start;
    }
    
    .txn-col-1,
    .txn-col-2 {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
    }
    
    /* Make Transaction Details column narrower */
    .txn-details-container {
        max-width: 100%;
    }
    
    /* Compact amount fields styling */
    .amount-fields-container {
        margin-top: 1rem;
        background: linear-gradient(135deg, #EFEBE9 0%, #D7CCC8 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #A1887F;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    
    .amount-fields-container .stNumberInput {
        font-size: 0.85rem;
    }
    
    .amount-fields-container label {
        font-size: 0.8rem !important;
    }
    
    /* Compact Transaction Details container - significantly reduced size */
    .compact-txn-container {
        background: linear-gradient(135deg, #EFEBE9 0%, #D7CCC8 100%);
        padding: 0.6rem 0.8rem;
        border-radius: 6px;
        border: 1.5px solid #A1887F;
        margin-bottom: 0.8rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        max-width: 90%;
    }
    
    .compact-txn-container h3 {
        font-size: 0.9rem !important;
        margin-bottom: 0.4rem !important;
        color: #5D4037;
        padding-bottom: 0.3rem;
        border-bottom: 1px solid #8D6E63;
    }
    
    .compact-txn-container .stSelectbox {
        margin-bottom: 0.2rem;
    }
    
    .compact-txn-container label {
        font-size: 0.65rem !important;
        margin-bottom: 0.15rem !important;
    }
    
    .compact-txn-container [data-baseweb="select"] {
        font-size: 0.7rem;
        padding: 0.25rem 0.4rem;
        min-height: 1.8rem;
    }
    
    .compact-txn-container [data-baseweb="input"] {
        font-size: 0.7rem;
        padding: 0.25rem 0.4rem;
    }
    
    /* Reduce column spacing in Transaction Details */
    .compact-txn-container [data-testid="column"] {
        padding: 0 0.2rem;
    }
    
    /* Make selectboxes in compact container smaller */
    .compact-txn-container .stSelectbox > div > div {
        padding: 0.2rem 0.3rem !important;
        min-height: 1.6rem !important;
    }
    
    /* Reduce overall width of Transaction Details container */
    .compact-txn-container {
        width: 85% !important;
        margin-left: auto;
        margin-right: auto;
    }
    
    
    /* Fraud alert */
    .fraud-alert {
        padding: 1.5rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #FFEBEE 0%, #FFCDD2 100%);
        border-left: 6px solid #D32F2F;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* Safe alert */
    .safe-alert {
        padding: 1.5rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 100%);
        border-left: 6px solid #388E3C;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* Explanation box */
    .explanation-box {
        padding: 1rem;
        background: linear-gradient(135deg, #FAFAFA 0%, #F5F5F5 100%);
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #8D6E63;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Prediction container - big box with distinct colors */
    .prediction-container {
        background: linear-gradient(135deg, #E8D5B7 0%, #D7CCC8 100%);
        padding: 2.5rem;
        border-radius: 20px;
        border: 4px solid #8D6E63;
        margin: 2rem 0;
        box-shadow: 0 8px 16px rgba(0,0,0,0.25);
        min-height: 400px;
        width: 100%;
    }
    
    /* Prediction metrics styling */
    .prediction-metric {
        background: linear-gradient(135deg, #FFF8E1 0%, #FFECB3 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 2px solid #8D6E63;
        text-align: center;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    
    /* Circular gauge chart styling */
    .gauge-container {
        position: relative;
        width: 200px;
        height: 200px;
        margin: 0 auto;
    }
    
    .gauge-svg {
        transform: rotate(-90deg);
    }
    
    .gauge-background {
        fill: none;
        stroke: #D7CCC8;
        stroke-width: 15;
    }
    
    .gauge-fill {
        fill: none;
        stroke-linecap: round;
        stroke-width: 15;
        transition: stroke-dasharray 0.5s;
    }
    
    .gauge-text {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        font-size: 2rem;
        font-weight: bold;
        color: #3E2723;
    }
    
    .gauge-label {
        position: absolute;
        bottom: -30px;
        left: 50%;
        transform: translateX(-50%);
        font-size: 0.9rem;
        color: #5D4037;
        font-weight: 500;
    }
    
    
    /* Header separator line - reduced margin */
    .header-separator {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent 0%, #8D6E63 20%, #8D6E63 80%, transparent 100%);
        margin: 1rem 0;
        width: 100%;
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model pipeline"""
    try:
        model = joblib.load("Model/rf_fraud_pipeline.pkl")
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

@st.cache_data
def load_feature_names():
    """Load the feature names used by the model"""
    try:
        feature_df = pd.read_csv("Data/rf_shap_feature_names.csv")
        return feature_df["feature_name"].values
    except Exception as e:
        st.error(f"Error loading feature names: {str(e)}")
        return None

@st.cache_resource
def create_shap_explainer(_model):
    """Create SHAP explainer for the model"""
    if not SHAP_AVAILABLE:
        return None
    try:
        # Get the preprocessor and model from pipeline
        preprocessor = _model.named_steps["preprocess"]
        rf_model = _model.named_steps["model"]
        
        # Create explainer
        explainer = shap.TreeExplainer(rf_model)
        return explainer
    except Exception as e:
        st.error(f"Error creating SHAP explainer: {str(e)}")
        return None

def clean_feature_name(raw_name):
    """Convert pipeline feature names into readable names"""
    name = raw_name.replace("cat__", "").replace("num__", "")
    name = name.replace("_", " ").lower()
    return name

# Feature explanations mapping (using lowercase keys to match cleaned names)
FEATURE_EXPLANATIONS = {
    "txn velocity 1h": "High number of transactions in the last hour",
    "txn velocity 24h": "High transaction activity in the last 24 hours",
    "velocity ratio": "Recent transaction spike compared to historical behavior",
    "ip risk score": "IP address is associated with high fraud risk",
    "device trust score": "Device has low trust history",
    "new device": "Transaction made from a new or unseen device",
    "amount usd": "Transaction amount is unusually high",
    "log amount usd": "Transaction amount deviates significantly from normal behavior",
    "fee ratio": "Transaction fee pattern is abnormal",
    "location mismatch": "IP location does not match expected user location",
    "kyc tier": "Low KYC verification level",
    "channel mobile": "Transaction performed via mobile channel",
    "channel web": "Transaction performed via web channel",
    "channel atm": "Transaction performed via ATM channel",
    "amount velocity interaction": "Combination of high amount and high velocity",
    "device ip risk": "Combined risk from device and IP address",
    "new device high velocity": "New device with high transaction velocity",
    "young account high amount": "New account with high transaction amount",
    "ip location risk": "High-risk IP with location mismatch",
    "risk score internal": "Internal risk assessment score",
    "account age days": "Account is relatively new",
    "chargeback history count": "History of chargebacks",
    "corridor risk": "High-risk transaction corridor",
    "fee": "Unusual fee amount",
    "is night": "Transaction occurred during unusual hours",
    "is weekend": "Transaction occurred on weekend",
    "new device velocity": "New device combined with high transaction velocity",
    "high risk device": "Device with high risk indicators",
    "amount usd capped": "Transaction amount (capped for analysis)",
    "log fee": "Fee amount (log transformed)",
    "ip usage count": "Number of transactions from this IP address",
    "day of week": "Day of the week when transaction occurred",
    "exchange rate src to dest": "Exchange rate between source and destination currencies",
    "amount src": "Transaction amount in source currency",
}

def shap_top_reasons_for_ui(model, explainer, X_row, feature_names, top_k=8):
    """Returns top SHAP reasons for fraud prediction (class 1)"""
    if not SHAP_AVAILABLE or explainer is None:
        return None
    try:
        # Transform input row exactly as the model sees it
        X_row_trans = model.named_steps["preprocess"].transform(X_row)
        
        # Get SHAP values
        exp = explainer(X_row_trans)
        
        # Extract SHAP values for fraud class (class index = 1)
        shap_vals = exp.values[0, :, 1]
        base_val = exp.base_values[0, 1]
        
        # Model predicted probability
        proba = model.predict_proba(X_row)[0, 1]
        
        # Top contributing features
        idx = np.argsort(np.abs(shap_vals))[::-1][:top_k]
        
        reasons = [
            {
                "feature": feature_names[j],
                "shap_value": float(shap_vals[j])
            }
            for j in idx
        ]
        
        return {
            "fraud_probability": float(proba),
            "base_value": float(base_val),
            "top_reasons": reasons
        }
    except Exception as e:
        st.error(f"Error computing SHAP values: {str(e)}")
        return None

def shap_reasons_to_text(ui_payload):
    """Convert SHAP reasons into readable text"""
    explanations = []
    
    for item in ui_payload["top_reasons"]:
        raw_feature = item["feature"]
        shap_val = item["shap_value"]
        
        clean_name = clean_feature_name(raw_feature)
        
        # Direction
        if shap_val > 0:
            direction = "increased"
            impact_type = "risk"
        else:
            direction = "reduced"
            impact_type = "safety"
        
        # Explanation text
        base_text = FEATURE_EXPLANATIONS.get(
            clean_name,
            f"{clean_name} influenced the fraud score"
        )
        
        # Capitalize first letter for display
        display_name = clean_name.title()
        
        explanations.append({
            "feature": display_name,
            "impact": direction,
            "strength": round(abs(shap_val), 4),
            "message": f"{base_text}, which {direction} the fraud risk.",
            "impact_type": impact_type
        })
    
    return explanations

def compute_derived_features(input_data):
    """Compute derived features from input data"""
    df = input_data.copy()
    
    # Velocity ratio
    df["velocity_ratio"] = df["txn_velocity_1h"] / (df["txn_velocity_24h"] + 1)
    
    # Fee ratio
    df["fee_ratio"] = df["fee"] / (df["amount_usd"] + 1e-6)  # Avoid division by zero
    
    # Amount velocity interaction
    df["amount_velocity_interaction"] = df["amount_usd"] * df["velocity_ratio"]
    
    # Device IP risk
    df["device_ip_risk"] = df["device_trust_score"] * df["ip_risk_score"]
    
    # New device velocity
    df["new_device_velocity"] = df["new_device"] * df["txn_velocity_1h"]
    
    # High risk device
    df["High risk device"] = df["new_device"].astype(int) * (1 - df["device_trust_score"])
    
    # Amount capped (99th percentile cap - using a reasonable default)
    df["amount_usd_capped"] = df["amount_usd"].clip(upper=df["amount_usd"].quantile(0.99) if len(df) > 1 else df["amount_usd"].max())
    
    # Log transforms
    df["log_amount_usd"] = np.log1p(df["amount_usd_capped"].clip(lower=0))
    df["log_fee"] = np.log1p(df["fee"].clip(lower=0))
    
    # New device high velocity
    df["new_device_high_velocity"] = ((df["new_device"] == 1) & (df["txn_velocity_1h"] >= 3)).astype(int)
    
    # Young account high amount
    df["young_account_high_amount"] = ((df["account_age_days"] < 30) & (df["amount_usd"] > 500)).astype(int)
    
    # IP location risk
    df["ip_location_risk"] = ((df["ip_risk_score"] > 0.7) & (df["location_mismatch"] == 1)).astype(int)
    
    # IP usage count (set to 1 as default since we don't have historical data)
    df["ip_usage_count"] = 1
    
    return df

def main():
    # Header without dark brown background box - compact and at absolute top
    # Main title - 3D plastic raised effect with all caps
    st.markdown('''
    <h1 style="
        color: #5D4037; 
        text-align: center; 
        font-size: 3.5rem; 
        font-weight: 900; 
        font-family: "Californian FB", "Californian", "Times New Roman", serif;
        margin-top: 0 !important; 
        margin-bottom: 0.1rem; 
        padding-top: 0 !important; 
        text-transform: uppercase;
        letter-spacing: 2px;
        text-shadow: 
            0 1px 0 #8D6E63,
            0 2px 0 #8D6E63,
            0 3px 1px rgba(0,0,0,.2),
            0 4px 2px rgba(0,0,0,.15);
    ">🔒 FRAUD DETECTION SYSTEM</h1>
    ''', unsafe_allow_html=True)
    
    # Subtitle - brown text, increased font size
    st.markdown('<p style="color: #6D4C41; text-align: center; font-size: 1.8rem; margin-top: 0.1rem; margin-bottom: 0.5rem; font-style: italic; padding: 0 1rem;">A machine learning-driven fraud prevention in cross-border financial services.</p>', unsafe_allow_html=True)
    
    # Feature highlights - 4 columns with smaller cards, reduced spacing
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="feature-card">⚡ Real-time Fraud Detection</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="feature-card">🤖 Explanability feature</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="feature-card">🎯 98% Accuracy</div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="feature-card">⚠️ Risk Assessment</div>', unsafe_allow_html=True)
    
    # Separator line after header with reduced margin
    st.markdown('<hr class="header-separator" style="margin: 0.2rem 0;">', unsafe_allow_html=True)
    
    # Load model and resources
    model = load_model()
    feature_names = load_feature_names()
    
    if model is None or feature_names is None:
        st.error("Failed to load model or feature names. Please check the file paths.")
        return
    
    explainer = create_shap_explainer(model)
    if explainer is None and SHAP_AVAILABLE:
        st.warning("SHAP explainer not available. Explanations will be limited.")
    
    # SIDEBAR: Transaction Information
    with st.sidebar:
        st.markdown("## 📝 Transaction Information")
        
        # Account Information - moved to top
        st.markdown("### 👤 Account Information")
        account_age_days = st.number_input("Account Age (Days)", min_value=0, value=100, step=1, key="account_age")
        kyc_tier = st.selectbox("KYC Tier", ["STANDARD", "ENHANCED", "LOW", "Not_Verified"], key="kyc_tier")
        chargeback_history_count = st.number_input("Chargeback History Count", min_value=0, value=0, step=1, key="chargeback")
        
        st.markdown("---")
        
        # Location (formerly Transaction Details) - moved to second position
        st.markdown("### 📍 Location")
        home_country = st.selectbox("Home Country", ["US", "CA", "UK", "Unknown"], key="home_country")
        source_currency = st.selectbox("Source Currency", ["USD", "CAD", "GBP"], key="source_currency")
        dest_currency = st.selectbox("Destination Currency", ["USD", "CAD", "GBP", "EUR", "CNY", "MXN", "INR", "NGN", "PHP"], key="dest_currency")
        channel = st.selectbox("Channel", ["WEB", "MOBILE", "ATM", "Unknown"], key="channel")
        ip_country = st.selectbox("IP Country", ["US", "CA", "UK", "Unknown"], key="ip_country")
        
        st.markdown("---")
        
        # Amount and fee fields
        st.markdown("### 💰 Amount & Fee")
        amount_usd = st.number_input("Amount (USD)", min_value=0.0, value=100.0, step=10.0, key="amount_usd")
        amount_src = st.number_input("Amount (Source Currency)", min_value=0.0, value=100.0, step=10.0, key="amount_src")
        fee = st.number_input("Fee", min_value=0.0, value=2.0, step=0.1, key="fee")
        exchange_rate_src_to_dest = st.number_input("Exchange Rate", min_value=0.0, value=1.0, step=0.01, key="exchange_rate")
        
        st.markdown("---")
        
        # Account and Device
        st.markdown("### 📱 Account and Device Activity")
        new_device = st.selectbox("New Device", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", key="new_device")
        location_mismatch = st.selectbox("Location Mismatch", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", key="location_mismatch")
        txn_velocity_1h = st.number_input("Transactions in Last 1 Hour", min_value=0, value=0, step=1, key="velocity_1h")
        txn_velocity_24h = st.number_input("Transactions in Last 24 Hours", min_value=0, value=0, step=1, key="velocity_24h")
        transaction_hour = st.number_input("Transaction Hour", min_value=0, max_value=23, value=12, step=1, key="transaction_hour")
        
        st.markdown("---")
        
        # Risk Indicators
        st.markdown("### ⚠️ Risk Indicators")
        ip_risk_score = st.slider("IP Risk Score", 0.0, 1.0, 0.3, 0.01, key="ip_risk")
        risk_score_internal = st.slider("Internal Risk Score", 0.0, 1.0, 0.3, 0.01, key="internal_risk")
        corridor_risk = st.slider("Corridor Risk", 0.0, 1.0, 0.0, 0.01, key="corridor_risk")
        device_trust_score = st.slider("Device Trust Score", 0.0, 1.0, 0.7, 0.01, key="device_trust")
        
        st.markdown("---")
        
        # Predict button
        predict_button = st.button("🔍 Analyze Transaction", type="primary", use_container_width=True)
    
    # Set default values for removed time features (always set these)
    day_of_week = 0
    is_weekend = 0
    is_night = 0
    # Map transaction hour (0-23) to time_of_day categories
    if 6 <= transaction_hour < 12:
        time_of_day = "morning"
    elif 12 <= transaction_hour < 18:
        time_of_day = "afternoon"
    elif 18 <= transaction_hour < 22:
        time_of_day = "evening"
    else:
        time_of_day = "late_night"
    
    # MAIN AREA: Prediction Results
    st.markdown('<h2 style="font-size: 1.5rem; margin-bottom: 1rem;">📊 Prediction Results</h2>', unsafe_allow_html=True)
    
    # Display placeholder or results
    if not predict_button:
        st.info("👈 Fill in the transaction details in the sidebar and click 'Analyze Transaction' to see results here.")
    
    # Handle prediction when button is clicked
    if predict_button:
        # Create input dataframe
        currency_pair = f"{source_currency}_{dest_currency}"
        
        input_data = pd.DataFrame({
            "home_country": [home_country],
            "source_currency": [source_currency],
            "dest_currency": [dest_currency],
            "channel": [channel],
            "amount_src": [amount_src],
            "amount_usd": [amount_usd],
            "fee": [fee],
            "exchange_rate_src_to_dest": [exchange_rate_src_to_dest],
            "new_device": [new_device],
            "ip_country": [ip_country],
            "location_mismatch": [location_mismatch],
            "ip_risk_score": [ip_risk_score],
            "kyc_tier": [kyc_tier],
            "account_age_days": [account_age_days],
            "device_trust_score": [device_trust_score],
            "chargeback_history_count": [chargeback_history_count],
            "risk_score_internal": [risk_score_internal],
            "txn_velocity_1h": [txn_velocity_1h],
            "txn_velocity_24h": [txn_velocity_24h],
            "corridor_risk": [corridor_risk],
            "day_of_week": [day_of_week],
            "is_weekend": [is_weekend],
            "is_night": [is_night],
            "time_of_day": [time_of_day],
            "currency_pair": [currency_pair]
        })
        
        # Compute derived features
        input_data = compute_derived_features(input_data)
        
        # Make prediction
        try:
            fraud_prob = model.predict_proba(input_data)[0, 1]
            fraud_prediction = model.predict(input_data)[0]
            
            # Display results in the main area - all metrics inside the container
            risk_level = "HIGH" if fraud_prob > 0.7 else "MEDIUM" if fraud_prob > 0.3 else "LOW"
            decision = "DECLINE" if fraud_prediction == 1 else "ALLOW"
            decision_color = "#D32F2F" if fraud_prediction == 1 else "#388E3C"
            prediction_text = "🚨 FRAUD" if fraud_prediction == 1 else "✅ LEGITIMATE"
            
            # Determine gauge color based on fraud probability
            if fraud_prob > 0.7:
                gauge_color = "#D32F2F"  # Red for high risk
            elif fraud_prob > 0.3:
                gauge_color = "#FF9800"  # Orange for medium risk
            else:
                gauge_color = "#388E3C"  # Green for low risk
            
            # Calculate circumference and stroke-dasharray for circular gauge
            radius = 85
            circumference = 2 * 3.14159 * radius
            stroke_dasharray = circumference
            stroke_dashoffset = circumference - (fraud_prob * circumference)
            
            st.markdown(f'''
            <div class="prediction-container">
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-bottom: 1.5rem;">
                    <div style="text-align: center; padding: 1rem;">
                        <h3 style="color: #5D4037; margin: 0 0 1rem 0; font-size: 1rem;">Fraud Probability</h3>
                        <div class="gauge-container">
                            <svg class="gauge-svg" width="200" height="200">
                                <circle class="gauge-background" cx="100" cy="100" r="{radius}"></circle>
                                <circle class="gauge-fill" cx="100" cy="100" r="{radius}" 
                                        stroke="{gauge_color}" 
                                        stroke-dasharray="{stroke_dasharray}" 
                                        stroke-dashoffset="{stroke_dashoffset}"></circle>
                            </svg>
                            <div class="gauge-text">{fraud_prob:.2%}</div>
                            <div class="gauge-label">Risk Level</div>
                        </div>
                    </div>
                    <div style="text-align: center; padding: 1rem;">
                        <h3 style="color: #5D4037; margin: 0 0 0.5rem 0; font-size: 1rem;">Prediction</h3>
                        <h2 style="color: #3E2723; margin: 0; font-size: 2rem; font-weight: bold;">{prediction_text}</h2>
                    </div>
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem;">
                    <div style="text-align: center; padding: 1rem;">
                        <h3 style="color: #5D4037; margin: 0 0 0.5rem 0; font-size: 1rem;">Risk Level</h3>
                        <h2 style="color: #3E2723; margin: 0; font-size: 2rem; font-weight: bold;">{risk_level}</h2>
                    </div>
                    <div style="text-align: center; padding: 1rem;">
                        <h3 style="color: {decision_color}; margin: 0 0 0.5rem 0; font-size: 1rem;">Decision</h3>
                        <h2 style="color: {decision_color}; margin: 0; font-size: 2rem; font-weight: bold;">{decision}</h2>
                    </div>
                </div>
            </div>
            ''', unsafe_allow_html=True)
            
            # Alert box
            if fraud_prediction == 1:
                st.markdown("""
                    <div class="fraud-alert">
                        <h3>⚠️ FRAUD DETECTED</h3>
                        <p>This transaction has been flagged as potentially fraudulent. Please review the explanations below.</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div class="safe-alert">
                        <h3>✅ Transaction Appears Legitimate</h3>
                        <p>This transaction shows no significant fraud indicators.</p>
                    </div>
                """, unsafe_allow_html=True)
            
            # SHAP Explanations - Only show for fraudulent transactions, Top 3 Risk-Increasing Factors
            if fraud_prediction == 1 and explainer is not None:  # Only show for fraud
                st.markdown("## 🔍 Explanation")
                st.markdown("The following factors contributed to this prediction:")
                
                # Get SHAP explanations - get more to filter for risk factors only
                ui_payload = shap_top_reasons_for_ui(model, explainer, input_data, feature_names, top_k=10)
                
                if ui_payload:
                    human_readable = shap_reasons_to_text(ui_payload)
                    
                    # Get only risk-increasing factors and take top 3
                    risk_factors = [e for e in human_readable if e["impact_type"] == "risk"][:3]
                    
                    if risk_factors:
                        st.markdown("### ⚠️ Fraud risk factor")
                        for i, factor in enumerate(risk_factors, 1):
                            st.markdown(f"""
                                <div class="explanation-box">
                                    <strong>{i}. {factor['feature']}</strong><br>
                                    {factor['message']}
                                </div>
                            """, unsafe_allow_html=True)
            elif fraud_prediction == 1 and explainer is None:
                st.warning("⚠️ SHAP explanations are not available. Please ensure SHAP is properly installed.")
            
            # Show input summary
            with st.expander("📋 View Transaction Summary"):
                st.dataframe(input_data.T, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.exception(e)
    

if __name__ == "__main__":
    main()

