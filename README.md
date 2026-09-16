# HELB Band Placement Predictor - Multinomial Logistic Regression

DSAIC Club demo project: predict which of 5 funding "bands" a Kenyan student
would be placed in, based on Means-Testing-Instrument-style socioeconomic
factors, using **multinomial logistic regression** (logistic regression
generalized from 2 classes to 5, via softmax instead of a single sigmoid).

**Important context:** HELB announced on **22 Aug 2025** that it dropped the
discrete 5-band system in favor of individualized, continuous need-scoring
for each student. This project recreates the earlier, well-documented band
system (used 2023-2025) because it's a clean, widely-relatable multi-class
classification example, not because bands are officially used today. If
you want to mirror the *current* system, reframe this as a regression
problem predicting a continuous "% of costs covered" instead (a nice link
back to the linear regression session).

**Note on the data:** `data/helb_band_placement.csv` is synthetically
generated (see `data/generate_data.py`) using the real, publicly reported
MTI factors: household income, parental occupation, orphan status,
disability, number of dependents, and place of residence. Band cutoffs are
illustrative approximations for teaching, not official HELB figures.

## Project structure
```
logistic/
├── data/
│   ├── generate_data.py            # builds the synthetic dataset
│   └── helb_band_placement.csv
├── model/
│   ├── helb_band_pipeline.joblib   # trained pipeline (preprocessing + model)
│   └── metrics.txt
├── plots/                          # EDA + evaluation charts
├── app/
│   └── app.py                      # Streamlit app
├── train.py                        # end-to-end training script
└── requirements.txt
```

## 1. Run locally
```bash
pip install -r requirements.txt
python3 train.py            # regenerates model/ and plots/
streamlit run app/app.py    # opens the demo in your browser
```

## 2. Deploy to Streamlit Community Cloud
1. Push this whole `logistic/` folder to a **public GitHub repo** (the repo
   root should contain `app/`, `model/`, `data/`, `requirements.txt`).
   ```bash
   git init
   git add .
   git commit -m "HELB band placement predictor"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<repo-name>.git
   git push -u origin main
   ```
2. Go to **share.streamlit.io** and sign in with GitHub.
3. Click **New app** → select your repo/branch → set **Main file path** to
   `app/app.py` → **Deploy**.
4. Streamlit Cloud installs `requirements.txt` and gives you a public URL
   within a couple of minutes.
5. Any future `git push` to `main` auto-redeploys the app.

## Model summary
Multinomial Logistic Regression, regularization strength `C` tuned with
5-fold cross-validated grid search on macro-F1 (macro, not accuracy, because
all 5 bands matter equally even though the model is easiest on the extremes).
See `model/metrics.txt` for the full per-band precision/recall/F1 on the
held-out test set. Band 1 (highest need) and Band 5 (lowest need) are
easiest to separate; the middle bands (2-4) are harder, as expected, since
need is really a continuum that has been cut into buckets.
