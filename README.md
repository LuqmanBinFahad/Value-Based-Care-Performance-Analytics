## Dr. Luqman Bin Fahad

A [Streamlit](https://luqmanbinfahad-value-based-care-performance-analytic-app-nz8m9f.streamlit.app/) prototype dashboard using **de-identified, real-world–inspired sample datasets** (patients, encounters, quality measures, and contracts). 
This is a working foundation you can swap with your own real-world datasets (CMS/ACO/EHR extracts).

## Features
- Overview KPIs (PMPM, admissions, ED, quality rates)
- Cost & Utilization trends
- Quality Measures tracking (filter by measure)
- Risk Stratification (HCC + comorbidity–based cohorts)
- Readmission Prediction (train-in-app logistic regression)
- High-Risk Cohort lists with export
- Contract Benchmarking & ROI

## How to run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Replace with real data
- `data/patients.csv`: patient-level features (age, sex, HCC, comorbidities, prior admits, LOS, PMPM, readmit_30d)
- `data/encounters.csv`: monthly utilization & cost (inpatient, ED, outpatient, total_cost)
- `data/quality_measures.csv`: HEDIS-like monthly numerators/denominators and rates
- `data/contracts.csv`: patient ACO attribution + benchmark PMPM

Ensure patient identifiers remain **de-identified** and comply with HIPAA when loading your datasets.
