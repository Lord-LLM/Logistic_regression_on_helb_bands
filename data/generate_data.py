"""
Synthetic HELB band-placement dataset for teaching purposes.

Modeled on the publicly reported Means Testing Instrument (MTI) factors used
by Kenya's Higher Education Loans Board (HELB) under the 2023-2025 New
Funding Model: household income, parental occupation, orphan status,
disability, number of dependents, and place-of-residence poverty level.

IMPORTANT CONTEXT: on 22 Aug 2025, HELB announced it had dropped the 5-band
system in favor of individualized, continuous need-scoring. This dataset
recreates the earlier band system (1 = highest need, 5 = lowest need)
because it's a clean, well-documented, widely-relatable multi-class example
-- not because bands are still officially used. Band cutoffs here are
illustrative approximations for teaching, not official HELB figures (only
the Band 1 income ceiling, ~KES 5,995/month, is a figure that has been
publicly reported).
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(21)
N = 1400

residence_types = ["rural_high_poverty", "rural_low_poverty", "urban_informal", "urban_formal"]
residence_weights = [0.30, 0.20, 0.25, 0.25]
# rough relative poverty probability index per residence type (higher = poorer area)
poverty_index_map = {"rural_high_poverty": 0.75, "rural_low_poverty": 0.45,
                      "urban_informal": 0.55, "urban_formal": 0.15}

parent_occupations = ["formal_employment", "informal_business", "unemployed", "deceased_both", "deceased_one"]
occupation_weights = [0.30, 0.38, 0.14, 0.06, 0.12]

residence = rng.choice(residence_types, size=N, p=residence_weights)
parent_occupation = rng.choice(parent_occupations, size=N, p=occupation_weights)

orphan_status = np.where(
    parent_occupation == "deceased_both", "double_orphan",
    np.where(parent_occupation == "deceased_one", "single_orphan", "not_orphan")
)

income_base_map = {"formal_employment": 55000, "informal_business": 22000,
                    "unemployed": 6000, "deceased_both": 3000, "deceased_one": 9000}
base_income = np.array([income_base_map[o] for o in parent_occupation])
poverty_factor = np.array([poverty_index_map[r] for r in residence])
household_monthly_income_kes = np.clip(
    rng.normal(base_income * (1 - 0.35 * poverty_factor), base_income * 0.30 + 1000),
    500, 300000
)

dependents_count = np.clip(rng.poisson(3.2), 0, 10)
siblings_in_college = np.clip(rng.poisson(0.6), 0, 4)
disability = rng.choice([0, 1], size=N, p=[0.93, 0.07])
gender = rng.choice(["female", "male"], size=N, p=[0.5, 0.5])

# ---- need score: higher = more need -> lower band number ----
need_score = (
    -0.9 * np.log1p(household_monthly_income_kes)      # lower income -> higher need
    + 0.35 * dependents_count
    + 0.25 * siblings_in_college
    + 1.4 * poverty_factor
    + 1.1 * disability
    + np.where(orphan_status == "double_orphan", 2.2, 0)
    + np.where(orphan_status == "single_orphan", 1.1, 0)
    + rng.normal(0, 0.5, size=N)
)

# bucket into 5 bands by quantiles of need_score: band 1 = highest need (top quantile)
quantiles = np.quantile(need_score, [0.2, 0.4, 0.6, 0.8])
band = np.digitize(need_score, quantiles, right=True)  # 0..4, 0 = lowest need_score
band = 5 - band  # flip so 1 = highest need, 5 = lowest need

df = pd.DataFrame({
    "household_monthly_income_kes": household_monthly_income_kes.round(0),
    "parent_occupation": parent_occupation,
    "orphan_status": orphan_status,
    "disability": disability,
    "dependents_count": dependents_count,
    "siblings_in_college": siblings_in_college,
    "residence_type": residence,
    "gender": gender,
    "band": band,
})

df.to_csv("data/helb_band_placement.csv", index=False)
print(df.shape)
print(df["band"].value_counts(normalize=True).sort_index())
print(df.groupby("band")["household_monthly_income_kes"].median())
