"""Week 2 pipeline: data collection, cleaning and preprocessing
Dataset: Supply Chain Analysis (Kaggle, harshsingh2209), file supply_chain_data.csv
Usage: python pipeline.py <input_csv> <output_dir>
"""
import sys
import os
import numpy as np
import pandas as pd

in_path, out_dir = sys.argv[1], sys.argv[2]
os.makedirs(out_dir, exist_ok=True)

# ---------------------------------------------------------------
# STEP 1: LOAD AND INSPECT (data collection)
# ---------------------------------------------------------------
df = pd.read_csv(in_path)
print("STEP 1  rows, columns:", df.shape)
print("        missing cells:", int(df.isna().sum().sum()))
print("        duplicate rows:", int(df.duplicated().sum()))

# ---------------------------------------------------------------
# STEP 2: STANDARDISE COLUMN NAMES AND TYPES
# ---------------------------------------------------------------
df.columns = (df.columns.str.strip().str.lower()
              .str.replace(" ", "_", regex=False))
text_cols = ["product_type", "customer_demographics", "shipping_carriers",
             "supplier_name", "location", "inspection_results",
             "transportation_modes", "routes"]
for c in text_cols:
    df[c] = df[c].str.strip().str.title().astype("category")
print("STEP 2  columns now:", list(df.columns)[:5], "...")

# ---------------------------------------------------------------
# STEP 3: LOGIC CHECKS ON THE REAL DATA
# ---------------------------------------------------------------
# 3a. Revenue should be roughly price x units sold
df["expected_revenue"] = df["price"] * df["number_of_products_sold"]
ratio = df["revenue_generated"] / df["expected_revenue"]
df["revenue_mismatch"] = ((ratio - 1).abs() > 0.5).astype(int)
print("STEP 3a revenue differs from price x sold by >50%:",
      int(df["revenue_mismatch"].sum()), "of", len(df))

# 3b. 'Unknown' customer group and 'Pending' inspection are placeholders
df["demographics_unknown"] = (df["customer_demographics"] == "Unknown").astype(int)
df["inspection_pending"] = (df["inspection_results"] == "Pending").astype(int)
print("STEP 3b unknown demographics:", int(df["demographics_unknown"].sum()),
      "| pending inspections:", int(df["inspection_pending"].sum()))

# 3c. Zero stock is a valid value (a stockout), so it is kept and flagged
df["stockout_flag"] = (df["stock_levels"] == 0).astype(int)
print("STEP 3c stockout rows:", int(df["stockout_flag"].sum()))

# ---------------------------------------------------------------
# STEP 4: MISSING VALUES AND OUTLIERS (reusable functions)
# ---------------------------------------------------------------
def iqr_bounds(s, k=1.5):
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr

def find_outliers(s):
    lo, hi = iqr_bounds(s)
    return (s < lo) | (s > hi)

num_cols = ["price", "availability", "number_of_products_sold",
            "revenue_generated", "stock_levels", "lead_times", "order_quantities",
            "shipping_times", "shipping_costs", "lead_time", "production_volumes",
            "manufacturing_lead_time", "manufacturing_costs", "defect_rates", "costs"]

real_outliers = {c: int(find_outliers(df[c]).sum()) for c in num_cols}
print("STEP 4  real outliers found (IQR):", sum(real_outliers.values()))

# ---------------------------------------------------------------
# STEP 5: SIMULATION - the real file is clean, so we damage a COPY
#         on purpose to show how each technique works
# ---------------------------------------------------------------
rng = np.random.default_rng(42)
sim = df[num_cols].copy()
truth = sim.copy()

miss_cols = ["price", "stock_levels", "shipping_costs"]
miss_idx = {}
for c in miss_cols:
    idx = rng.choice(sim.index, size=5, replace=False)   # 5 of 100 rows = 5%
    sim.loc[idx, c] = np.nan
    miss_idx[c] = idx

out_idx = {}
for c, factor in [("shipping_times", None), ("costs", None)]:
    idx = rng.choice(sim.index.difference(
        np.concatenate(list(miss_idx.values()))), size=3, replace=False)
    out_idx[c] = idx
sim.loc[out_idx["shipping_times"], "shipping_times"] = [60, 75, 90]   # days
sim.loc[out_idx["costs"], "costs"] = sim.loc[out_idx["costs"], "costs"] * 10

print("\nSTEP 5  SIMULATION (injected problems)")
print("        missing injected:", {c: int(sim[c].isna().sum()) for c in miss_cols})
print("        outliers injected: shipping_times=3, costs=3")

# 5a. Detect
print("        detected missing:", int(sim.isna().sum().sum()))
det = {c: int(find_outliers(sim[c].dropna()).sum()) for c in ["shipping_times", "costs"]}
print("        detected outliers (IQR):", det)
zdet = {c: int((((sim[c] - sim[c].mean()) / sim[c].std()).abs() > 3).sum())
        for c in ["shipping_times", "costs"]}
print("        detected outliers (|z|>3):", zdet)

# 5b. Fix missing values with the median, compare with the mean
fixed = sim.copy()
imputed_err = {}
for c in miss_cols:
    med = sim[c].median()
    mean_ = sim[c].mean()
    idx = miss_idx[c]
    err_med = float((truth.loc[idx, c] - med).abs().mean())
    err_mean = float((truth.loc[idx, c] - mean_).abs().mean())
    imputed_err[c] = (round(err_med, 2), round(err_mean, 2))
    fixed[c] = sim[c].fillna(med)
print("        mean abs error of fill (median, mean):", imputed_err)
print("        missing after fix:", int(fixed.isna().sum().sum()))

# 5c. Fix outliers by capping (winsorising) at the IQR bounds
for c in ["shipping_times", "costs"]:
    lo, hi = iqr_bounds(sim[c])
    fixed[c] = fixed[c].clip(lower=lo, upper=hi)
    print(f"        {c}: bounds=({lo:.2f}, {hi:.2f}) "
          f"max before={sim[c].max():.2f} after={fixed[c].max():.2f}")

# 5d. Effect on the average (before fix, after fix, true value)
print("\n        EFFECT ON MEAN (damaged -> fixed vs true)")
rows = []
for c in ["price", "stock_levels", "shipping_costs", "shipping_times", "costs"]:
    rows.append((c, round(sim[c].mean(), 2), round(fixed[c].mean(), 2), round(truth[c].mean(), 2),
                 round(sim[c].std(), 2), round(fixed[c].std(), 2), round(truth[c].std(), 2)))
    print("       ", rows[-1])

# ---------------------------------------------------------------
# STEP 6: NORMALISATION (put numbers on a similar scale)
# ---------------------------------------------------------------
scale_cols = ["price", "shipping_costs", "costs", "lead_times",
              "production_volumes", "defect_rates"]
minmax = (df[scale_cols] - df[scale_cols].min()) / (df[scale_cols].max() - df[scale_cols].min())
zscore = (df[scale_cols] - df[scale_cols].mean()) / df[scale_cols].std()
print("\nSTEP 6  NORMALISATION (real data)")
print("        BEFORE ranges:")
for c in scale_cols:
    print(f"          {c}: {df[c].min():.2f} to {df[c].max():.2f}")
print("        AFTER min-max: min", minmax.min().round(2).unique(), "max", minmax.max().round(2).unique())
print("        AFTER z-score means:", zscore.mean().round(2).unique(), "std:", zscore.std().round(2).unique())

# robustness check: what one big outlier does to each method
s = sim["costs"]
mm_dmg = (s - s.min()) / (s.max() - s.min())
print("        min-max on damaged 'costs': median of scaled values =", round(float(mm_dmg.median()), 3),
      "(squeezed toward 0 by the outliers)")
mm_fix = (fixed["costs"] - fixed["costs"].min()) / (fixed["costs"].max() - fixed["costs"].min())
print("        min-max on fixed   'costs': median of scaled values =", round(float(mm_fix.median()), 3))

# ---------------------------------------------------------------
# STEP 7: SAVE CLEAN FILE
# ---------------------------------------------------------------
clean = df.copy()
for c in scale_cols:
    clean[c + "_scaled"] = minmax[c]
clean.to_csv(os.path.join(out_dir, "supply_chain_clean.csv"), index=False)
print("\nSTEP 7  saved supply_chain_clean.csv with", clean.shape[0], "rows and", clean.shape[1], "columns")
