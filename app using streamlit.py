"""
Customer Churn Risk Dashboard
Interactive Streamlit app: explore the scored customer base, browse risk
tiers, and get a live churn-probability prediction for any hypothetical
customer profile via the "What-If Predictor" tab.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

Deploy free:
    Push this repo to GitHub, then deploy at https://share.streamlit.io
    pointing at app.py (no extra config needed).
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance

# ----------------------------------------------------------------------
# PAGE CONFIG & STYLE
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Churn Risk Dashboard",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

NAVY = "#1F3A5F"
ACCENT = "#E0623B"
TIER_COLORS = {"Low": "#5C9C5C", "Medium": "#E0B23B", "High": "#E0823B", "Critical": "#C0392B"}

st.markdown(f"""
<style>
    .stApp {{ background-color: #FAFBFC; }}
    div[data-testid="stMetric"] {{
        background-color: white;
        border: 1px solid #E5E9EE;
        border-radius: 10px;
        padding: 14px 18px;
    }}
    div[data-testid="stMetricLabel"] {{ color: #5A6B7B; }}
    h1, h2, h3 {{ color: {NAVY}; }}
    .stTabs [data-baseweb="tab"] {{ font-weight: 600; }}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------
# DATA LOADING & MODEL TRAINING (cached so it only runs once per session)
# ----------------------------------------------------------------------
PRETTY_NAMES = {
    "monthly_fee": "Monthly Fee",
    "avg_weekly_usage_hours": "Avg Weekly Usage (hrs)",
    "support_tickets": "Support Tickets",
    "payment_failures": "Payment Failures",
    "tenure_months": "Tenure (months)",
    "last_login_days_ago": "Days Since Last Login",
    "usage_per_dollar": "Usage per Dollar Spent",
    "plan_type_Premium": "Plan: Premium",
    "plan_type_Standard": "Plan: Standard",
}

FEATURE_COLS_NUMERIC = ["monthly_fee", "avg_weekly_usage_hours", "support_tickets",
                         "payment_failures", "tenure_months", "last_login_days_ago",
                         "usage_per_dollar"]


def risk_tier(p):
    if p >= 0.75:
        return "Critical"
    elif p >= 0.5:
        return "High"
    elif p >= 0.25:
        return "Medium"
    return "Low"


@st.cache_data
def load_data():
    df = pd.read_csv("data/customer_subscription_churn_usage_patterns.csv")
    df["signup_date"] = pd.to_datetime(df["signup_date"])
    df["churn_flag"] = (df["churn"] == "Yes").astype(int)
    df["usage_per_dollar"] = df["avg_weekly_usage_hours"] / (df["monthly_fee"] / 100)
    return df


@st.cache_resource
def train_model(df):
    X = pd.get_dummies(df[FEATURE_COLS_NUMERIC + ["plan_type"]], columns=["plan_type"], drop_first=True)
    y = df["churn_flag"]
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = GradientBoostingClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)
    test_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    perm = permutation_importance(model, X_test, y_test, n_repeats=20, random_state=42,
                                   scoring="roc_auc", n_jobs=-1)
    imp_df = pd.DataFrame({
        "feature": feature_names,
        "importance": perm.importances_mean,
    }).sort_values("importance", ascending=False)
    imp_df["pretty"] = imp_df["feature"].map(lambda f: PRETTY_NAMES.get(f, f))

    # score the full dataset with the trained model
    full_proba = model.predict_proba(X)[:, 1]

    return model, feature_names, test_auc, imp_df, full_proba


def build_whatif_row(feature_names, plan_type, monthly_fee, usage_hrs, tickets,
                      failures, tenure, last_login):
    usage_per_dollar = usage_hrs / (monthly_fee / 100)
    row = {
        "monthly_fee": monthly_fee,
        "avg_weekly_usage_hours": usage_hrs,
        "support_tickets": tickets,
        "payment_failures": failures,
        "tenure_months": tenure,
        "last_login_days_ago": last_login,
        "usage_per_dollar": usage_per_dollar,
        "plan_type_Premium": 1 if plan_type == "Premium" else 0,
        "plan_type_Standard": 1 if plan_type == "Standard" else 0,
    }
    return pd.DataFrame([row])[feature_names]


df = load_data()
model, feature_names, test_auc, imp_df, full_proba = train_model(df)

df_scored = df.copy()
df_scored["churn_risk_score"] = full_proba
df_scored["risk_tier"] = df_scored["churn_risk_score"].apply(risk_tier)

# ----------------------------------------------------------------------
# SIDEBAR FILTERS
# ----------------------------------------------------------------------
st.sidebar.title("📉 Churn Risk Dashboard")
st.sidebar.caption("Filter the customer base below. Filters apply to the Overview and Customer Explorer tabs.")

tier_filter = st.sidebar.multiselect(
    "Risk tier", options=["Low", "Medium", "High", "Critical"],
    default=["Low", "Medium", "High", "Critical"],
)
plan_filter = st.sidebar.multiselect(
    "Plan type", options=sorted(df_scored["plan_type"].unique()),
    default=sorted(df_scored["plan_type"].unique()),
)
search_id = st.sidebar.text_input("Search by user ID")

st.sidebar.divider()
st.sidebar.markdown(f"**Model:** Gradient Boosting  \n**Test AUC-ROC:** {test_auc:.3f}")
st.sidebar.caption("Trained live on app startup using the same pipeline as the project notebook. Practice dataset — see project README for details.")

filtered = df_scored[
    df_scored["risk_tier"].isin(tier_filter) & df_scored["plan_type"].isin(plan_filter)
]
if search_id:
    filtered = filtered[filtered["user_id"].astype(str).str.contains(search_id)]

# ----------------------------------------------------------------------
# HEADER + KPIs
# ----------------------------------------------------------------------
st.title("Customer Churn Risk Dashboard")
st.caption("Subscription business churn prediction — live model, scored customer base, and a what-if risk predictor.")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Customers (filtered)", f"{len(filtered):,}", help="Updates with sidebar filters")
k2.metric("Avg. churn risk", f"{filtered['churn_risk_score'].mean()*100:.1f}%" if len(filtered) else "—")
k3.metric("Critical-tier customers", f"{(filtered['risk_tier']=='Critical').sum():,}")
k4.metric("Model AUC-ROC (test set)", f"{test_auc:.3f}")

st.divider()

# ----------------------------------------------------------------------
# TABS
# ----------------------------------------------------------------------
tab_overview, tab_explorer, tab_whatif = st.tabs(["📊 Overview", "🔍 Customer Explorer", "🎛️ What-If Predictor"])

# ---- TAB 1: OVERVIEW ----
with tab_overview:
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Risk Tier Distribution")
        tier_counts = filtered["risk_tier"].value_counts().reindex(["Low", "Medium", "High", "Critical"]).fillna(0)
        fig = px.bar(
            x=tier_counts.index, y=tier_counts.values,
            color=tier_counts.index, color_discrete_map=TIER_COLORS,
            labels={"x": "Risk tier", "y": "Customers"},
        )
        fig.update_layout(showlegend=False, height=380)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Churn Rate by Plan Type")
        plan_churn = filtered.groupby("plan_type")["churn_flag"].mean().sort_values(ascending=False) * 100
        fig2 = px.bar(
            x=plan_churn.index, y=plan_churn.values,
            labels={"x": "Plan", "y": "Historical churn rate (%)"},
            color_discrete_sequence=[ACCENT],
        )
        fig2.update_layout(height=380)
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Key Churn Drivers (Permutation Importance)")
    fig3 = px.bar(
        imp_df.sort_values("importance"), x="importance", y="pretty", orientation="h",
        labels={"importance": "Importance (drop in AUC-ROC)", "pretty": ""},
        color_discrete_sequence=[ACCENT],
    )
    fig3.update_layout(height=420)
    st.plotly_chart(fig3, use_container_width=True)
    st.caption("Days since last login and payment failures are the strongest predictors — plan type and price barely matter.")

# ---- TAB 2: CUSTOMER EXPLORER ----
with tab_explorer:
    st.subheader(f"Scored Customers ({len(filtered):,} matching filters)")
    sort_col = st.selectbox(
        "Sort by", ["churn_risk_score", "last_login_days_ago", "payment_failures",
                     "support_tickets", "tenure_months"],
        format_func=lambda c: PRETTY_NAMES.get(c, c.replace("_", " ").title()),
    )
    display_df = filtered.sort_values(sort_col, ascending=False)[
        ["user_id", "plan_type", "monthly_fee", "avg_weekly_usage_hours", "support_tickets",
         "payment_failures", "tenure_months", "last_login_days_ago", "churn_risk_score", "risk_tier"]
    ].rename(columns={"churn_risk_score": "risk_score"})
    display_df["risk_score"] = (display_df["risk_score"] * 100).round(1)

    st.dataframe(
        display_df, use_container_width=True, height=480,
        column_config={
            "risk_score": st.column_config.ProgressColumn(
                "Risk score (%)", min_value=0, max_value=100, format="%.1f%%"
            ),
        },
    )

    csv_bytes = display_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download filtered list as CSV", csv_bytes,
                        file_name="filtered_churn_risk.csv", mime="text/csv")

# ---- TAB 3: WHAT-IF PREDICTOR ----
with tab_whatif:
    st.subheader("Score a Hypothetical Customer")
    st.caption("Adjust the sliders to see how the model's predicted churn risk changes in real time.")

    c1, c2 = st.columns(2)
    with c1:
        plan_type = st.selectbox("Plan type", ["Basic", "Standard", "Premium"], index=1)
        monthly_fee = {"Basic": 199, "Standard": 399, "Premium": 699}[plan_type]
        st.metric("Monthly fee", f"₹{monthly_fee}")
        usage_hrs = st.slider("Avg weekly usage (hours)", 0.0, 30.0, 10.0, 0.5)
        tenure = st.slider("Tenure (months)", 1, 36, 12)
    with c2:
        tickets = st.slider("Support tickets", 0, 10, 2)
        failures = st.slider("Payment failures", 0, 6, 0)
        last_login = st.slider("Days since last login", 0, 60, 10)

    whatif_row = build_whatif_row(feature_names, plan_type, monthly_fee, usage_hrs,
                                   tickets, failures, tenure, last_login)
    proba = float(model.predict_proba(whatif_row)[:, 1][0])
    tier = risk_tier(proba)

    st.divider()
    g1, g2 = st.columns([1, 1.4])
    with g1:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=proba * 100,
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": TIER_COLORS[tier]},
                "steps": [
                    {"range": [0, 25], "color": "#EAF3EA"},
                    {"range": [25, 50], "color": "#FBF3E0"},
                    {"range": [50, 75], "color": "#FBE7DB"},
                    {"range": [75, 100], "color": "#F8DAD5"},
                ],
            },
            title={"text": "Predicted churn probability"},
        ))
        fig_gauge.update_layout(height=320, margin=dict(t=60, b=10, l=20, r=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with g2:
        st.markdown(f"### Risk tier: <span style='color:{TIER_COLORS[tier]}'>{tier}</span>", unsafe_allow_html=True)
        action_map = {
            "Critical": "Immediate outreach — personal call or high-value retention offer within 48 hours.",
            "High": "Proactive email/in-app nudge with an engagement incentive within 1 week.",
            "Medium": "Add to nurture campaign; monitor for risk-score increases.",
            "Low": "No action needed — healthy profile.",
        }
        st.info(action_map[tier])
        st.markdown("**Profile summary**")
        st.write(
            f"- Plan: **{plan_type}** (₹{monthly_fee}/mo)\n"
            f"- Usage: **{usage_hrs} hrs/week**\n"
            f"- Support tickets: **{tickets}**\n"
            f"- Payment failures: **{failures}**\n"
            f"- Tenure: **{tenure} months**\n"
            f"- Last login: **{last_login} days ago**"
        )

st.divider()
st.caption(
    "Built as a churn-prediction portfolio project. Model trained live on a synthetic/practice dataset "
    "(see README for details). Source: github repo — notebooks/churn_prediction_analysis.ipynb for the full analysis."
)
