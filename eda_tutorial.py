"""Step-by-step exploratory data analysis tutorial.

The script creates a reproducible synthetic telecom churn dataset if the CSV is
not present, then performs EDA and saves tables and charts under outputs/eda_results.
The generated data is for practice only; it does not describe real customers.
"""

# Step 1: import packages and set paths
from pathlib import Path
import numpy as np
import pandas as pd
from html import escape

BASE = Path(__file__).resolve().parent
DATA_FILE = BASE / "telecom_churn_synthetic.csv"
RESULTS = BASE / "eda_results"
RESULTS.mkdir(exist_ok=True)


# Step 2: create a practice dataset (skip this block if using your own CSV)
def create_practice_dataset(path: Path, rows: int = 1000) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    customer_id = [f"C{i:05d}" for i in range(1, rows + 1)]
    tenure = rng.integers(1, 73, rows)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], rows,
                          p=[0.55, 0.25, 0.20])
    internet = rng.choice(["Fiber optic", "DSL", "No internet"], rows,
                          p=[0.50, 0.35, 0.15])
    support_calls = np.clip(rng.poisson(1.4, rows), 0, 8)
    monthly_charges = np.where(internet == "Fiber optic",
                               rng.normal(85, 12, rows),
                               np.where(internet == "DSL", rng.normal(57, 9, rows),
                                        rng.normal(25, 6, rows)))
    monthly_charges = np.clip(monthly_charges, 15, 125).round(2)
    autopay = rng.choice(["Yes", "No"], rows, p=[0.58, 0.42])
    # Churn probability is intentionally associated with short tenure,
    # month-to-month plans, support calls, and lack of autopay.
    logit = (-2.0 + 0.035 * (24 - tenure)
             + 0.95 * (contract == "Month-to-month")
             + 0.32 * support_calls
             + 0.55 * (autopay == "No")
             + 0.012 * (monthly_charges - 65))
    probability = 1 / (1 + np.exp(-logit))
    churn = np.where(rng.random(rows) < probability, "Yes", "No")
    df = pd.DataFrame({
        "customer_id": customer_id,
        "tenure_months": tenure,
        "contract_type": contract,
        "internet_service": internet,
        "support_calls_90d": support_calls,
        "monthly_charges": monthly_charges,
        "autopay": autopay,
        "churn": churn,
    })
    # Add a small amount of realistic missingness for cleaning practice.
    for col, rate in [("monthly_charges", 0.025), ("support_calls_90d", 0.015)]:
        df.loc[rng.choice(rows, size=round(rows * rate), replace=False), col] = np.nan
    df.to_csv(path, index=False)
    return df


if not DATA_FILE.exists():
    df = create_practice_dataset(DATA_FILE)
else:
    df = pd.read_csv(DATA_FILE)

# Step 3: inspect dataset dimensions, sample records, and types
print("STEP 3 — DATASET OVERVIEW")
print(f"Rows: {df.shape[0]:,} | Columns: {df.shape[1]}")
print(df.head())
print("\nColumn types:\n", df.dtypes)

# Step 4: assess completeness, duplicates, and unique values
quality = pd.DataFrame({
    "missing_count": df.isna().sum(),
    "missing_percent": (df.isna().mean() * 100).round(2),
    "unique_values": df.nunique(dropna=False),
})
print("\nSTEP 4 — DATA QUALITY\n", quality)
print("Duplicate rows:", df.duplicated().sum())
quality.to_csv(RESULTS / "data_quality.csv")

# Step 5: calculate descriptive statistics
print("\nSTEP 5 — DESCRIPTIVE STATISTICS")
print(df.describe(include="all").T)
df.describe(include="all").T.to_csv(RESULTS / "summary_statistics.csv")

# Step 6: inspect target balance (important before classification)
target_counts = df["churn"].value_counts(dropna=False)
target_rate = (df["churn"].eq("Yes").mean() * 100).round(2)
print("\nSTEP 6 — TARGET BALANCE\n", target_counts)
print(f"Churn rate: {target_rate}%")

# Step 7: compare churn by customer groups
group_cols = ["contract_type", "autopay", "internet_service"]
group_rates = {}
for col in group_cols:
    group_rates[col] = (df.groupby(col, dropna=False)["churn"]
                        .apply(lambda s: s.eq("Yes").mean() * 100)
                        .sort_values(ascending=False).round(1))
    print(f"\nChurn rate (%) by {col}:\n{group_rates[col]}")

# Step 8: compare numeric features by churn status
numeric_comparison = (df.groupby("churn")[
    ["tenure_months", "support_calls_90d", "monthly_charges"]]
    .mean().round(2))
print("\nNumeric means by churn status:\n", numeric_comparison)
numeric_comparison.to_csv(RESULTS / "numeric_means_by_churn.csv")

# Step 9: create labeled SVG bar charts (no plotting package required)
def save_bar_chart(title, categories, values, y_label, filename, value_suffix=""):
    width, height = 760, 420
    left, right, top, bottom = 90, 30, 70, 105
    chart_w, chart_h = width - left - right, height - top - bottom
    max_value = max(values) if values and max(values) > 0 else 1
    slot = chart_w / max(1, len(categories))
    bars = []
    for i, (category, value) in enumerate(zip(categories, values)):
        bar_w = slot * 0.58
        bar_h = chart_h * value / max_value
        x = left + i * slot + (slot - bar_w) / 2
        y = top + chart_h - bar_h
        bars.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="#59788e"/>'
            f'<text x="{x + bar_w/2:.1f}" y="{y - 9:.1f}" text-anchor="middle" font-size="14">{value:.1f}{value_suffix}</text>'
            f'<text x="{x + bar_w/2:.1f}" y="{top + chart_h + 25}" text-anchor="middle" font-size="13">{escape(str(category))}</text>'
        )
    ticks = []
    for tick in range(5):
        val = max_value * tick / 4
        y = top + chart_h - chart_h * tick / 4
        ticks.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#dddddd"/>')
        ticks.append(f'<text x="{left-12}" y="{y+5:.1f}" text-anchor="end" font-size="12">{val:.0f}</text>')
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
           f'<rect width="100%" height="100%" fill="white"/><text x="{width/2}" y="32" text-anchor="middle" font-size="20" font-family="Arial">{escape(title)}</text>'
           + ''.join(ticks) + ''.join(bars)
           + f'<text x="22" y="{height/2}" transform="rotate(-90 22 {height/2})" text-anchor="middle" font-size="14">{escape(y_label)}</text>'
           + f'<text x="{width/2}" y="{height-22}" text-anchor="middle" font-size="14">Category</text></svg>')
    (RESULTS / filename).write_text(svg, encoding="utf-8")

counts = df["churn"].value_counts().reindex(["No", "Yes"], fill_value=0)
save_bar_chart("Customer count by churn status", counts.index.tolist(), counts.tolist(),
               "Customers (count)", "churn_counts.svg")
contract_plot = df.groupby("contract_type")["churn"].apply(
    lambda s: s.eq("Yes").mean() * 100).sort_values(ascending=False)
save_bar_chart("Churn rate by contract type", contract_plot.index.tolist(), contract_plot.tolist(),
               "Churn rate (%)", "churn_by_contract.svg", "%")
autopay_plot = df.groupby("autopay")["churn"].apply(
    lambda s: s.eq("Yes").mean() * 100).reindex(["Yes", "No"])
save_bar_chart("Churn rate by autopay enrollment", autopay_plot.index.tolist(), autopay_plot.tolist(),
               "Churn rate (%)", "churn_by_autopay.svg", "%")

# Step 10: save a cleaned copy for modeling practice.
# Median fills the numeric fields here; in real model evaluation, fit imputation
# on training data only to avoid leakage from validation/test data.
model_df = df.copy()
for col in ["monthly_charges", "support_calls_90d"]:
    model_df[col] = model_df[col].fillna(model_df[col].median())
model_df.to_csv(RESULTS / "practice_data_cleaned.csv", index=False)
print(f"\nOutputs saved to: {RESULTS}")
