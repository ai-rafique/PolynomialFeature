import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Ad Sales Predictor",
    page_icon="📣",
    layout="wide"
)

# ── Load & train model (cached so it only runs once) ─────────────────
@st.cache_resource
def load_and_train():
    df = pd.read_csv("advertising.csv")
    X = df[["TV", "Radio", "Newspaper"]]
    y = df["Sales"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = Pipeline([
        ("poly",  PolynomialFeatures(degree=2, include_bias=False)),
        ("reg",   LinearRegression())
    ])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = {
        "R²":   round(r2_score(y_test, y_pred), 4),
        "MAE":  round(mean_absolute_error(y_test, y_pred), 4),
        "RMSE": round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
    }
    return model, df, X_test, y_test, y_pred, metrics

model, df, X_test, y_test, y_pred, metrics = load_and_train()

# ── Header ───────────────────────────────────────────────────────────
st.title("📣 Advertising Sales Predictor")
st.markdown("Polynomial Regression (Degree 2) · King County Advertising Dataset · 200 observations")
st.divider()

# ── Sidebar — budget inputs ──────────────────────────────────────────
st.sidebar.header("🎯 Set Your Ad Budget")
st.sidebar.markdown("Adjust each channel's spend (in $K) to get an instant sales prediction.")

tv        = st.sidebar.slider("📺 TV Budget ($K)",        0.0, 300.0, 150.0, step=1.0)
radio     = st.sidebar.slider("📻 Radio Budget ($K)",     0.0,  50.0,  30.0, step=0.5)
newspaper = st.sidebar.slider("📰 Newspaper Budget ($K)", 0.0, 115.0,  20.0, step=0.5)
total_budget = tv + radio + newspaper

sample = pd.DataFrame([[tv, radio, newspaper]], columns=["TV", "Radio", "Newspaper"])
pred   = model.predict(sample)[0]

# ── Prediction banner ────────────────────────────────────────────────
st.subheader("💰 Sales Prediction")
col1, col2, col3, col4 = st.columns(4)
col1.metric("📺 TV Spend",         f"${tv:.1f}K")
col2.metric("📻 Radio Spend",      f"${radio:.1f}K")
col3.metric("📰 Newspaper Spend",  f"${newspaper:.1f}K")
col4.metric("🛒 Predicted Sales",  f"${pred:.2f}K",
            delta=f"Total budget: ${total_budget:.1f}K")

# Estimated ROI
roi = ((pred / total_budget) * 100) if total_budget > 0 else 0
st.markdown(f"**Estimated ROI:** Every $1K spent returns approximately **${pred/total_budget:.2f}K** in sales."
            if total_budget > 0 else "")

st.divider()

# ── Model metrics row ────────────────────────────────────────────────
st.subheader("📊 Model Performance")
m1, m2, m3 = st.columns(3)
m1.metric("R² Score",  metrics["R²"],  help="Proportion of variance explained (closer to 1 = better)")
m2.metric("MAE ($K)",  metrics["MAE"], help="Average prediction error in $K")
m3.metric("RMSE ($K)", metrics["RMSE"],help="Root mean squared error in $K")

st.divider()

# ── Plots row ────────────────────────────────────────────────────────
st.subheader("📈 Visual Analysis")
tab1, tab2, tab3 = st.tabs(["Actual vs Predicted", "Diminishing Returns", "Budget Breakdown"])

with tab1:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].scatter(y_test, y_pred, alpha=0.6, color="#4C8CBF", s=40)
    lims = [min(float(y_test.min()), float(y_pred.min())) - 1,
            max(float(y_test.max()), float(y_pred.max())) + 1]
    axes[0].plot(lims, lims, "r--", linewidth=1.5, label="Perfect fit")
    axes[0].set_xlabel("Actual Sales ($K)")
    axes[0].set_ylabel("Predicted Sales ($K)")
    axes[0].set_title("Actual vs Predicted", fontweight="bold")
    axes[0].legend()

    residuals = np.array(y_test) - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.6, color="#E07B3A", s=40)
    axes[1].axhline(0, color="red", linestyle="--", linewidth=1.5)
    axes[1].set_xlabel("Predicted Sales ($K)")
    axes[1].set_ylabel("Residual")
    axes[1].set_title("Residuals vs Fitted", fontweight="bold")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with tab2:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    channels = ["TV", "Radio", "Newspaper"]
    colors   = ["#4C8CBF", "#E07B3A", "#3BAA74"]
    tv_m, rad_m, news_m = df["TV"].mean(), df["Radio"].mean(), df["Newspaper"].mean()

    for ax, ch, c in zip(axes, channels, colors):
        x_range = np.linspace(0, df[ch].max(), 300)
        if ch == "TV":
            X_sim = pd.DataFrame({"TV": x_range, "Radio": rad_m,  "Newspaper": news_m})
        elif ch == "Radio":
            X_sim = pd.DataFrame({"TV": tv_m,    "Radio": x_range, "Newspaper": news_m})
        else:
            X_sim = pd.DataFrame({"TV": tv_m,    "Radio": rad_m,  "Newspaper": x_range})

        y_sim = model.predict(X_sim)
        ax.plot(x_range, y_sim, color=c, linewidth=2.5)
        ax.scatter(df[ch], df["Sales"], alpha=0.15, s=15, color=c)

        # Mark current slider value
        if ch == "TV":     cur = tv
        elif ch == "Radio": cur = radio
        else:              cur = newspaper
        cur_pred = model.predict(pd.DataFrame([[tv, radio, newspaper]], columns=["TV","Radio","Newspaper"]))[0]
        ax.axvline(cur, color="black", linestyle=":", linewidth=1.5, label=f"Your ${cur:.0f}K")
        ax.legend(fontsize=8)
        ax.set_xlabel(f"{ch} Budget ($K)")
        ax.set_ylabel("Predicted Sales ($K)")
        ax.set_title(f"{ch}: Returns Curve", fontweight="bold")
        ax.grid(alpha=0.3)

    plt.suptitle("Diminishing Returns — Others Held at Dataset Mean", fontsize=12, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    st.caption("The vertical dotted line shows your current slider position on each channel's returns curve.")

with tab3:
    if total_budget > 0:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        # Budget pie
        sizes  = [tv, radio, newspaper]
        labels = [f"TV\n${tv:.0f}K", f"Radio\n${radio:.0f}K", f"Newspaper\n${newspaper:.0f}K"]
        clrs   = ["#4C8CBF", "#E07B3A", "#3BAA74"]
        axes[0].pie([s if s > 0 else 0.001 for s in sizes], labels=labels,
                    colors=clrs, autopct="%1.1f%%", startangle=90,
                    wedgeprops={"edgecolor": "white", "linewidth": 1.5})
        axes[0].set_title("Budget Allocation", fontweight="bold")

        # Bar comparison: linear vs poly prediction
        lin_model = Pipeline([("poly", PolynomialFeatures(degree=1, include_bias=False)),
                               ("reg",  LinearRegression())])
        lin_model.fit(df[["TV","Radio","Newspaper"]], df["Sales"])
        lin_pred  = lin_model.predict(sample)[0]

        axes[1].bar(["Linear\n(Degree 1)", "Polynomial\n(Degree 2)"],
                    [lin_pred, pred],
                    color=["#aaaaaa", "#4C8CBF"], edgecolor="white", width=0.5)
        axes[1].set_ylabel("Predicted Sales ($K)")
        axes[1].set_title("Linear vs Polynomial Prediction\nfor Your Budget", fontweight="bold")
        for i, v in enumerate([lin_pred, pred]):
            axes[1].text(i, v + 0.1, f"${v:.2f}K", ha="center", fontweight="bold")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    else:
        st.info("Set at least one budget slider above zero to see breakdown charts.")

st.divider()

# ── Diminishing returns insight ───────────────────────────────────────
st.subheader("💡 Insights & Recommendations")

col_a, col_b = st.columns(2)
with col_a:
    st.markdown("**Channel Effectiveness at Your Budget:**")
    # Marginal return: predict with +1K on each channel
    base = model.predict(sample)[0]
    for ch, cur_val in zip(["TV", "Radio", "Newspaper"], [tv, radio, newspaper]):
        bump = sample.copy()
        bump[ch] = cur_val + 1
        marginal = model.predict(bump)[0] - base
        emoji = "🟢" if marginal > 0.05 else ("🟡" if marginal > 0 else "🔴")
        st.markdown(f"{emoji} **{ch}**: adding $1K more → **{marginal:+.3f}K** in sales")

with col_b:
    st.markdown("**Budget Allocation Tip:**")
    alloc_pct = {ch: round(v / total_budget * 100, 1) if total_budget > 0 else 0
                 for ch, v in zip(["TV", "Radio", "Newspaper"], [tv, radio, newspaper])}
    if alloc_pct.get("Newspaper", 0) > 25:
        st.warning("📰 Newspaper spend is high (>25% of budget). Consider reallocating to TV or Radio for better returns.")
    if alloc_pct.get("TV", 0) > 75:
        st.info("📺 TV dominates your budget. A mix with Radio may unlock interaction synergies.")
    if tv > 200:
        st.warning("📺 TV spend is in the high-diminishing-returns zone (>$200K). Marginal gains are low.")
    if alloc_pct.get("Radio", 0) > 0 and alloc_pct.get("TV", 0) > 0:
        st.success("✅ Using both TV and Radio together — these channels have a positive interaction effect.")

st.divider()
st.caption("Model: Polynomial Regression Degree 2 | Dataset: Advertising.csv (200 obs) | R² = 0.9533")