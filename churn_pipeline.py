"""
Customer Churn Prediction Pipeline
Subscription business: predicts churn risk and identifies key drivers.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (roc_auc_score, roc_curve, classification_report,
                              confusion_matrix, precision_recall_curve, average_precision_score)
from sklearn.inspection import permutation_importance

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)
COLOR_CHURN = "#E0623B"
COLOR_RETAIN = "#3B7EA1"
PALETTE2 = [COLOR_RETAIN, COLOR_CHURN]

CHART_DIR = "images"

# ---------------------------------------------------------------
# 1. LOAD
# ---------------------------------------------------------------
df = pd.read_csv("data/customer_subscription_churn_usage_patterns.csv")
df["signup_date"] = pd.to_datetime(df["signup_date"])
df["churn_flag"] = (df["churn"] == "Yes").astype(int)

print("Rows:", len(df))
print(df["churn"].value_counts(normalize=True))

# ---------------------------------------------------------------
# 2. EDA CHARTS
# ---------------------------------------------------------------

# 2a. Overall churn rate
fig, ax = plt.subplots(figsize=(5, 4))
counts = df["churn"].value_counts()
ax.pie(counts, labels=["Churned" if l == "Yes" else "Retained" for l in counts.index],
       autopct="%1.1f%%", colors=[COLOR_CHURN if l == "Yes" else COLOR_RETAIN for l in counts.index],
       startangle=90, wedgeprops={"edgecolor": "white", "linewidth": 2})
ax.set_title("Overall Churn Rate", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/01_overall_churn_rate.png", dpi=150)
plt.close()

# 2b. Churn rate by plan type
plan_churn = df.groupby("plan_type")["churn_flag"].mean().sort_values(ascending=False) * 100
fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(plan_churn.index, plan_churn.values, color=COLOR_CHURN, width=0.55)
ax.set_ylabel("Churn rate (%)")
ax.set_title("Churn Rate by Plan Type", fontsize=13, fontweight="bold")
for b in bars:
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1, f"{b.get_height():.1f}%",
            ha="center", fontsize=10, fontweight="bold")
ax.set_ylim(0, max(plan_churn.values) + 12)
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/02_churn_by_plan.png", dpi=150)
plt.close()

# 2c. Distributions of key numeric features by churn status
num_feats = ["avg_weekly_usage_hours", "support_tickets", "payment_failures",
             "tenure_months", "last_login_days_ago"]
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()
for i, feat in enumerate(num_feats):
    sns.kdeplot(data=df, x=feat, hue="churn", fill=True, alpha=0.45,
                palette={"No": COLOR_RETAIN, "Yes": COLOR_CHURN}, ax=axes[i], common_norm=False, legend=(i==0))
    axes[i].set_title(feat.replace("_", " ").title(), fontsize=11, fontweight="bold")
    axes[i].set_xlabel("")
axes[-1].axis("off")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/03_feature_distributions.png", dpi=150)
plt.close()

# 2d. Correlation heatmap
corr_cols = num_feats + ["monthly_fee", "churn_flag"]
corr = df[corr_cols].corr()
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1,
            square=True, cbar_kws={"shrink": .8}, ax=ax)
ax.set_title("Feature Correlation Matrix", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/04_correlation_heatmap.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# 3. FEATURE ENGINEERING
# ---------------------------------------------------------------
df["usage_per_dollar"] = df["avg_weekly_usage_hours"] / (df["monthly_fee"] / 100)
df["is_inactive_30d"] = (df["last_login_days_ago"] > 30).astype(int)
df["had_payment_failure"] = (df["payment_failures"] > 0).astype(int)
df["high_support_load"] = (df["support_tickets"] >= df["support_tickets"].median()).astype(int)
df["tenure_bucket"] = pd.cut(df["tenure_months"], bins=[0, 6, 12, 24, 100],
                              labels=["0-6mo", "6-12mo", "12-24mo", "24mo+"])

feature_cols_numeric = ["monthly_fee", "avg_weekly_usage_hours", "support_tickets",
                         "payment_failures", "tenure_months", "last_login_days_ago",
                         "usage_per_dollar"]
feature_cols_categorical = ["plan_type"]

X = pd.get_dummies(df[feature_cols_numeric + feature_cols_categorical], columns=feature_cols_categorical, drop_first=True)
y = df["churn_flag"]
feature_names = X.columns.tolist()

# ---------------------------------------------------------------
# 4. TRAIN / TEST SPLIT
# ---------------------------------------------------------------
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, df.index, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------
# 5. MODELS
# ---------------------------------------------------------------
log_reg = LogisticRegression(max_iter=2000, random_state=42)
log_reg.fit(X_train_scaled, y_train)

rf = RandomForestClassifier(n_estimators=400, max_depth=6, min_samples_leaf=10,
                             random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

gb = GradientBoostingClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42)
gb.fit(X_train, y_train)

models = {"Logistic Regression": (log_reg, X_test_scaled), "Random Forest": (rf, X_test),
          "Gradient Boosting": (gb, X_test)}

results = {}
for name, (model, X_te) in models.items():
    proba = model.predict_proba(X_te)[:, 1]
    pred = model.predict(X_te)
    auc = roc_auc_score(y_test, proba)
    ap = average_precision_score(y_test, proba)
    results[name] = {"model": model, "proba": proba, "pred": pred, "auc": auc, "ap": ap}
    print(f"\n=== {name} ===")
    print(f"AUC-ROC: {auc:.4f}  | Avg Precision: {ap:.4f}")
    print(classification_report(y_test, pred, target_names=["Retained", "Churned"]))

# pick best model by AUC
best_name = max(results, key=lambda n: results[n]["auc"])
best = results[best_name]
print(f"\nBest model: {best_name} (AUC={best['auc']:.4f})")

# ---------------------------------------------------------------
# 6. ROC + PR CURVES (all models)
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
colors = {"Logistic Regression": "#3B7EA1", "Random Forest": "#5C9C5C", "Gradient Boosting": "#E0623B"}
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res["proba"])
    axes[0].plot(fpr, tpr, label=f"{name} (AUC={res['auc']:.3f})", color=colors[name], linewidth=2)
axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4)
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC Curve", fontweight="bold")
axes[0].legend(loc="lower right", fontsize=9)

for name, res in results.items():
    prec, rec, _ = precision_recall_curve(y_test, res["proba"])
    axes[1].plot(rec, prec, label=f"{name} (AP={res['ap']:.3f})", color=colors[name], linewidth=2)
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].set_title("Precision-Recall Curve", fontweight="bold")
axes[1].legend(loc="lower left", fontsize=9)
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/05_roc_pr_curves.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# 7. CONFUSION MATRIX (best model)
# ---------------------------------------------------------------
cm = confusion_matrix(y_test, best["pred"])
fig, ax = plt.subplots(figsize=(5, 4.5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Retained", "Churned"], yticklabels=["Retained", "Churned"], ax=ax,
            annot_kws={"size": 14, "fontweight": "bold"})
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title(f"Confusion Matrix — {best_name}", fontweight="bold")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/06_confusion_matrix.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# 8. FEATURE IMPORTANCE (permutation importance on best model, model-agnostic)
# ---------------------------------------------------------------
X_te_for_perm = X_test_scaled if best_name == "Logistic Regression" else X_test
perm = permutation_importance(best["model"], X_te_for_perm, y_test, n_repeats=30,
                               random_state=42, scoring="roc_auc", n_jobs=-1)
imp_df = pd.DataFrame({
    "feature": feature_names,
    "importance": perm.importances_mean,
    "std": perm.importances_std
}).sort_values("importance", ascending=False)

pretty_names = {
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
imp_df["pretty"] = imp_df["feature"].map(lambda f: pretty_names.get(f, f))

fig, ax = plt.subplots(figsize=(8, 5))
imp_plot = imp_df.sort_values("importance")
bars = ax.barh(imp_plot["pretty"], imp_plot["importance"], xerr=imp_plot["std"],
                color=COLOR_CHURN, alpha=0.85, error_kw={"alpha": 0.4})
ax.set_xlabel("Permutation Importance (drop in AUC-ROC)")
ax.set_title(f"Key Churn Drivers — {best_name}", fontweight="bold", fontsize=13)
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/07_feature_importance.png", dpi=150)
plt.close()

print("\nTop drivers:")
print(imp_df[["pretty", "importance"]].to_string(index=False))

# ---------------------------------------------------------------
# 9. LOGISTIC REGRESSION COEFFICIENTS (direction of effect)
# ---------------------------------------------------------------
coef_df = pd.DataFrame({
    "feature": feature_names,
    "coef": log_reg.coef_[0]
}).sort_values("coef")
coef_df["pretty"] = coef_df["feature"].map(lambda f: pretty_names.get(f, f))

fig, ax = plt.subplots(figsize=(8, 5))
colors_dir = [COLOR_CHURN if c > 0 else COLOR_RETAIN for c in coef_df["coef"]]
ax.barh(coef_df["pretty"], coef_df["coef"], color=colors_dir)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Standardized Coefficient (log-odds of churn)")
ax.set_title("Direction of Effect on Churn (Logistic Regression)", fontweight="bold", fontsize=12)
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/08_coefficient_direction.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# 10. SCORE ENTIRE CUSTOMER BASE + RISK TIERS
# ---------------------------------------------------------------
X_full_scaled = scaler.transform(X) if best_name == "Logistic Regression" else X
all_proba = best["model"].predict_proba(X_full_scaled if best_name == "Logistic Regression" else X)[:, 1]

df_scored = df.copy()
df_scored["churn_risk_score"] = all_proba

def risk_tier(p):
    if p >= 0.75:
        return "Critical"
    elif p >= 0.5:
        return "High"
    elif p >= 0.25:
        return "Medium"
    else:
        return "Low"

df_scored["risk_tier"] = df_scored["churn_risk_score"].apply(risk_tier)

# Risk tier distribution chart
tier_order = ["Low", "Medium", "High", "Critical"]
tier_counts = df_scored["risk_tier"].value_counts().reindex(tier_order)
fig, ax = plt.subplots(figsize=(6, 4))
tier_colors = ["#5C9C5C", "#E0B23B", "#E0823B", "#C0392B"]
bars = ax.bar(tier_counts.index, tier_counts.values, color=tier_colors)
ax.set_ylabel("Number of customers")
ax.set_title("Customer Base by Churn Risk Tier", fontweight="bold")
for b in bars:
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 5, str(int(b.get_height())),
            ha="center", fontsize=10, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/09_risk_tier_distribution.png", dpi=150)
plt.close()

out_cols = ["user_id", "signup_date", "plan_type", "monthly_fee", "avg_weekly_usage_hours",
            "support_tickets", "payment_failures", "tenure_months", "last_login_days_ago",
            "churn", "churn_risk_score", "risk_tier"]
df_scored[out_cols].sort_values("churn_risk_score", ascending=False).to_csv(
    "outputs/scored_customers_churn_risk.csv", index=False)

# Save summary stats to a text file for report-building
with open("outputs/summary_stats.txt", "w") as f:
    f.write(f"N_ROWS={len(df)}\n")
    f.write(f"OVERALL_CHURN_RATE={df['churn_flag'].mean():.4f}\n")
    for name, res in results.items():
        f.write(f"{name}_AUC={res['auc']:.4f}\n")
        f.write(f"{name}_AP={res['ap']:.4f}\n")
    f.write(f"BEST_MODEL={best_name}\n")
    f.write(f"BEST_AUC={best['auc']:.4f}\n")
    cr = classification_report(y_test, best["pred"], target_names=["Retained", "Churned"], output_dict=True)
    f.write(f"BEST_PRECISION_CHURN={cr['Churned']['precision']:.4f}\n")
    f.write(f"BEST_RECALL_CHURN={cr['Churned']['recall']:.4f}\n")
    f.write(f"BEST_F1_CHURN={cr['Churned']['f1-score']:.4f}\n")
    for _, row in plan_churn.items():
        pass
    f.write("PLAN_CHURN_RATES=" + str(plan_churn.round(1).to_dict()) + "\n")
    f.write("TOP5_DRIVERS=" + str(imp_df.head(5)[["pretty", "importance"]].values.tolist()) + "\n")
    f.write("TIER_COUNTS=" + str(tier_counts.to_dict()) + "\n")
    f.write("TIER_PCT=" + str((tier_counts / len(df_scored) * 100).round(1).to_dict()) + "\n")

print("\nPipeline complete. Charts in", CHART_DIR, "| outputs in outputs")
