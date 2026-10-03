# Telecom churn EDA practice package

This is a **synthetic practice dataset** generated with a fixed random seed. It contains no real customer records. The churn labels were constructed to have relationships with tenure, contract type, support calls, monthly charges, and autopay; the results below are therefore examples for learning, not evidence about a real telecom business.

## Run it step by step

1. Install Python and the two packages used by the script: `pip install pandas numpy`.
2. Keep `eda_tutorial.py` and `telecom_churn_synthetic.csv` in the same folder.
3. Run `python eda_tutorial.py`. If the CSV is absent, the script creates it first.
4. Read the console output: shape/types, quality checks, descriptive statistics, target balance, group churn rates, and numeric comparisons.
5. Open the three SVG charts in `eda_results/`; the script also saves summary CSVs and an imputed practice copy there.
6. For a different dataset, replace the CSV and update the target/feature column names in the script. Keep the original customer identifier out of model features.

## Findings from the included 1,000-row sample

- **Target balance:** 268 customers churned and 732 did not, giving a 26.8% churn rate. Use stratified train/validation splits and report precision, recall, F1, and PR-AUC alongside accuracy.
- **Contract type:** churn was 32.7% for month-to-month, 17.4% for one-year, and 21.5% for two-year contracts. Contract type may be a useful predictor and a segment for retention experiments; test whether the pattern holds on real data.
- **Tenure:** churners averaged 27.12 months of tenure versus 40.25 months among non-churners. Check churn by tenure band and consider nonlinear effects rather than assuming a straight-line relationship.
- **Support contacts:** churners averaged 1.63 recent support calls versus 1.25 for non-churners. Support-call count may reflect unresolved issues; investigate timing and reason codes before deciding on an intervention.
- **Monthly charges:** churners averaged 69.10 versus 65.30 among non-churners. The difference is modest and could be confounded by internet service and contract mix, so compare within segments.
- **Data quality:** monthly charges are missing for 2.5% of rows and support calls for 1.5%; there are no duplicate rows. Fit imputers on training data only and preserve missingness indicators if missingness itself may carry information.
- **Autopay:** the observed rates are 28.6% without autopay and 25.4% with autopay. This small gap is not evidence that enabling autopay causes lower churn; it may be useful as a predictor, but a randomized test is needed for a causal retention claim.

## Columns

`customer_id` is an identifier; `tenure_months` is account tenure; `contract_type`, `internet_service`, and `autopay` are categories; `support_calls_90d` is recent support-call count; `monthly_charges` is a synthetic monthly amount; `churn` is the Yes/No label.
