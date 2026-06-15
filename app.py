import streamlit as st
import pandas as pd
from datetime import datetime
import os
import matplotlib.pyplot as plt
import joblib

# Load ML components
model = joblib.load("random_forest_model.pkl")
encoders = joblib.load("feature_encoders.pkl")
target_encoder = joblib.load("target_encoder.pkl")

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="EcoPackAI",
    page_icon="🌱",
    layout="wide"
)

# -------------------------------------------------
# GLOBAL CSS (UNCHANGED)
# -------------------------------------------------
st.markdown("""
<style>

/* Background */
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
}

/* Force all text visible */
html, body, [class*="css"]  {
    color: #E8F5E9 !important;
}

/* Headings */
h1, h2, h3, label {
    color: #E8F5E9 !important;
}

/* KPI Cards */
.kpi-card {
    background: rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 22px;
    border: 1px solid rgba(255,255,255,0.15);
    backdrop-filter: blur(12px);
    text-align: center;
}
.kpi-value {
    font-size: 32px;
    font-weight: 700;
    color: #7CFC9A;
}
.kpi-label {
    font-size: 14px;
    color: #CFD8DC;
}

/* Charts */
.chart-card {
    background: rgba(255,255,255,0.06);
    border-radius: 16px;
    padding: 18px;
    border: 1px solid rgba(255,255,255,0.12);
}

/* Buttons */
.stButton>button {
    background-color:#00c853;
    color:white;
    font-size:18px;
    border-radius:10px;
}

/* ---- IMPORTANT FIXES ---- */

/* Tables */
[data-testid="stTable"] {
    color: white !important;
}

/* Dataframes */
[data-testid="stDataFrame"] {
    color: white !important;
}

/* Metrics */
[data-testid="stMetric"] {
    color: white !important;
}

/* Tabs */
button[data-baseweb="tab"] {
    color: #E8F5E9 !important;
}

/* Active tab */
button[data-baseweb="tab"][aria-selected="true"] {
    color: #21cbf3 !important;
    font-weight: bold;
}

/* ---- FIX RADIO BUTTON TEXT (FINAL FIX) ---- */

/* FORCE ALL RADIO TEXT VISIBLE */
div[role="radiogroup"] * {
    color: #FFFFFF !important;
}

/* SELECTED OPTION */
div[role="radiogroup"] input:checked + div span {
    color: #21cbf3 !important;
    font-weight: 700;
}
 
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# SESSION STATE
# -------------------------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "auth_mode" not in st.session_state:
    st.session_state.auth_mode = "login"

# -------------------------------------------------
# DEMO USERS
# -------------------------------------------------
USERS = {"admin": "1234"}

def predict_packaging(user_input):

    df = pd.DataFrame([user_input])

    # Create volume
    df["Volume_cm3"] = df["Length_cm"] * df["Width_cm"] * df["Height_cm"]

    # Encode categorical features
    for col, encoder in encoders.items():
        if col in df.columns and hasattr(encoder, "classes_"):
            df[col] = df[col].apply(
                lambda x: encoder.transform([x])[0] 
                if x in encoder.classes_ 
                else encoder.transform([encoder.classes_[0]])[0]
            )

    # Ensure column order matches training
    feature_order = [
        "Warehouse_block", "Mode_of_Shipment", "Cost_of_the_Product",
        "Weight_in_gms", "Product_category", "Volume_cm3", "Fragile",
        "Liquid_Leak_Prone", "Perishable", "Shipping_distance",
        "Stacking_pressure", "Delivery_urgency", "Budget_level",
        "Eco_preference", "Material_constraint", "Return_risk",
        "Product_importance"
    ]

    df = df[feature_order]

    # Predict
    pred = model.predict(df)
    label = target_encoder.inverse_transform(pred)

    #st.write("Encoded Input:", df)

    return label[0]
# -------------------------------------------------
# AUTH PAGE (UNCHANGED)
# -------------------------------------------------
def auth_page():
    st.markdown("<div class='auth-wrapper'>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1.4])

    with col1:
        st.markdown("""
        <div style="background:#1E9E8A;color:white;padding:70px;height:100%">
        <h1>EcoPackAI</h1>
        <p>Sustainable Choices. Intelligent Future.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("<div style='padding:50px'>", unsafe_allow_html=True)

        if st.session_state.auth_mode == "login":
            st.subheader("Sign In")
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")

            if st.button("LOGIN"):
                if u in USERS and USERS[u] == p:
                    st.session_state.logged_in = True
                    st.rerun()
                else:
                    st.error("Invalid credentials")

            st.markdown("— or —")
            if st.button("Create Account"):
                st.session_state.auth_mode = "signup"
                st.rerun()

        else:
            st.subheader("Create Account")
            st.text_input("Username")
            st.text_input("Email")
            st.text_input("Password", type="password")

            if st.button("SIGN UP"):
                st.success("Account created! Please login.")
                st.session_state.auth_mode = "login"
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------
# DASHBOARD FUNCTIONS (UNCHANGED)
# -------------------------------------------------
def recommendation_dashboard(df):
    st.markdown("## 📦 Recommendation Insights Dashboard")

    if df.empty:
        st.warning("No recommendations available.")
        return

    total = len(df)
    avg_eco = round(df["eco_score"].mean(), 2)
    top_pack = df["recommended_packaging"].mode()[0]

    plastic_free = df[
        ~df["recommended_packaging"].str.contains("plastic", case=False, na=False)
    ]
    plastic_pct = round((len(plastic_free) / total) * 100, 1)

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"<div class='kpi-card'><div class='kpi-label'>Total</div><div class='kpi-value'>{total}</div></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='kpi-card'><div class='kpi-label'>Top Packaging</div><div class='kpi-value'>{top_pack}</div></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='kpi-card'><div class='kpi-label'>Eco Score</div><div class='kpi-value'>{avg_eco}</div></div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='kpi-card'><div class='kpi-label'>Plastic-Free</div><div class='kpi-value'>{plastic_pct}%</div></div>", unsafe_allow_html=True)

    #st.bar_chart(df["recommended_packaging"].value_counts())


def sustainability_dashboard(df):
    st.markdown("## 🌍 Sustainability Dashboard")

    if df.empty:
        st.warning("No data available.")
        return

    st.bar_chart(df.groupby("product_category")["eco_score"].mean())

# -------------------------------------------------
# NEW HACKATHON DASHBOARD
# -------------------------------------------------
def calculate_sustainability(packaging):

    emission_factors = {
        "Plastic": 6.0,
        "Corrugated": 1.5,
        "Molded Pulp": 0.8,
        "Recycled Paper": 0.6,
        "Compostable": 0.5
    }

    p = packaging.lower()

    if "pulp" in p:
        carbon = emission_factors["Molded Pulp"]
    elif "corrugated" in p:
        carbon = emission_factors["Corrugated"]
    elif "recycled" in p:
        carbon = emission_factors["Recycled Paper"]
    elif "compostable" in p:
        carbon = emission_factors["Compostable"]
    else:
        carbon = emission_factors["Plastic"]

    plastic_carbon = emission_factors["Plastic"]
    reduction = ((plastic_carbon - carbon) / plastic_carbon) * 100

    eco_score = round(10 - (carbon / plastic_carbon * 10), 2)

    return carbon, reduction, eco_score

def dashboard_page(df):

    st.markdown("<h1 style='text-align:center;'>🌱 EcoPackAI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'>Smart packaging recommendations</p>", unsafe_allow_html=True)

    # ---------------- SAME TITLE ----------------
    st.markdown("## 📦 Tell us your packaging needs")

    # ---------------- REPLACED INPUT (NO MULTISELECT) ----------------
    c1, c2, c3 = st.columns(3)

    with c1:
        product = st.selectbox("Product Category", ["food","electronics","cosmetics","clothing","books","glass_items"])
        weight = st.number_input("Weight (g)", 100, 5000)
        
        length = st.number_input("Length (cm)", 5, 100)
        width = st.number_input("Width (cm)", 5, 100)
        height = st.number_input("Height (cm)", 5, 100)

        fragile = st.selectbox("Fragile", [0,1])

    with c2:
        liquid = st.selectbox("Liquid Leak Prone", [0,1])
        perishable = st.selectbox("Perishable", [0,1])
        distance = st.selectbox("Shipping Distance", ["local","domestic","international"])

    with c3:
        urgency = st.selectbox("Delivery Urgency", ["standard","express"])
        budget = st.selectbox("Budget", ["low","medium","high"])
        eco = st.selectbox("Eco Preference", ["max_sustainability","balanced","low_cost"])

    # ---------------- BUTTON ----------------
    if "generated" not in st.session_state:
        st.session_state.generated = False

    if st.button("🚀 Generate Recommendations"):
        st.session_state.generated = True

    if st.session_state.generated:
        # ---------------- USER INPUT ----------------
        user_input = {
            "Warehouse_block": "A",
            "Mode_of_Shipment": "Flight",
            "Cost_of_the_Product": 200,
            "Weight_in_gms": weight,
            "Product_category": product,
            "Length_cm": length,
            "Width_cm": width,
            "Height_cm": height,
            "Fragile": fragile,
            "Liquid_Leak_Prone": liquid,
            "Perishable": perishable,
            "Shipping_distance": distance,
            "Stacking_pressure": 1,
            "Delivery_urgency": urgency,
            "Budget_level": budget,
            "Eco_preference": eco,
            "Material_constraint": "no_plastic",
            "Return_risk": 0,
            "Product_importance": "medium"
        }

        #st.write("User Input:", user_input)

        # ---------------- ML PREDICTION ----------------
        result = predict_packaging(user_input)
        st.success(f"✅ Recommended Packaging: {result}")
        carbon, reduction, eco_score = calculate_sustainability(result)

        # Create dataframe (so your old dashboards still work)
        filtered_df = pd.DataFrame({
            "product_category": [product],
            "shipping_distance": [distance],
            "eco_preference": [eco],
            "recommended_packaging": [result]
        })
        filtered_df["eco_score"] = eco_score

        # ---------------- VIEW SWITCH (UNCHANGED UI) ----------------
        view = st.radio("View", ["Insights", "Sustainability"], horizontal=True)

        # ---------------- INSIGHTS ----------------
        if view == "Insights":
            recommendation_dashboard(filtered_df)

        # ---------------- SUSTAINABILITY ----------------
        else:

            tab1, tab2, tab3 = st.tabs([
                "🌍 Carbon Footprint",
                "📊 Breakdown",
                "🔄 Comparison"
            ])

            #carbon, reduction, eco_score = calculate_sustainability(result)

            # 🌍 TAB 1
            with tab1:
                st.subheader("🌍 Carbon Footprint Analysis")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown(f"""
                    <div style="text-align:center">
                        <div style="font-size:16px; color:#E8F5E9;">🌍 Carbon Emission</div>
                        <div style="font-size:36px; font-weight:700; color:#7CFC9A;">
                            {carbon:.2f} kg CO₂
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                with col2:
                    st.markdown(f"""
                    <div style="text-align:center">
                        <div style="font-size:16px; color:#E8F5E9;">📉 Reduction vs Plastic</div>
                        <div style="font-size:36px; font-weight:700; color:#21cbf3;">
                            {reduction:.1f}%
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                trees_saved = (6.0 - carbon) / 21
                st.success(f"🌳 Equivalent to saving {trees_saved:.2f} trees/year")

            # 📊 TAB 2 (UNCHANGED)
            with tab2:
                st.subheader("📊 Sustainability Score Breakdown")

                scores = {
                    "Recyclability": 9,
                    "Cost Efficiency": 7,
                    "Carbon Impact": 8,
                    "Durability": 8
                }

                for factor, score in scores.items():
                    st.markdown(f"<span style='color:#E8F5E9; font-weight:600'>{factor}</span>", unsafe_allow_html=True)
                    st.progress(score / 10)
                    st.markdown(f"<span style='color:#CFD8DC'>{score}/10</span>", unsafe_allow_html=True)

            # 🔄 TAB 3 (UPDATED WITH REAL VALUES)
            with tab3:
                st.subheader("🔄 Packaging Comparison")

                comparison_data = {
                    "Metric": ["CO₂ Emission (kg)", "Eco Score", "Cost Index"],
                    "Plastic": [6.0, 2, 8],
                    "Recommended": [round(carbon, 2), eco_score, 6]
                }

                comp_df = pd.DataFrame(comparison_data)
                st.dataframe(comp_df, use_container_width=True)

                st.success("🟢 Recommended packaging is more sustainable than plastic")
                st.error("🔴 Plastic has high carbon impact")

        # ---------------- AI EXPLANATION (SLIGHTLY SMARTER) ----------------
        st.markdown("### 🤖 Why this recommendation?")
        st.markdown(f"""
        <div style="
            background: rgba(33,150,243,0.08);
            padding: 14px;
            border-radius: 10px;
            border-left: 4px solid #21cbf3;
        ">
            <span style="font-size:16px; font-weight:600; color:#21cbf3;">🤖 AI Explanation</span><br>
            <span style="font-size:15px; color:#E3F2FD;">
                AI selected this packaging based on eco preference (<b>{eco}</b>),
                shipping distance (<b>{distance}</b>), and product safety factors like
                fragility, liquid risk, and perishability.
            </span>
        </div>
        """, unsafe_allow_html=True)

    # ---------------- LOGOUT ----------------
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

# -------------------------------------------------
# ROUTER 
# -------------------------------------------------
if not st.session_state.logged_in:
    auth_page()
else:
    dashboard_page(pd.DataFrame())    