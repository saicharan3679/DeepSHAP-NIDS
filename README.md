# Concept-Drift-Resilient NIDS

A runnable research/educational simulator inspired by:

M. Komarchesqui et al., "A Comprehensive Survey on Concept-Drift-Resilient Network Intrusion Detection Systems," IEEE Access, 2026. DOI: 10.1109/ACCESS.2026.3691262.

## What the project demonstrates

- Non-stationary network-like traffic
- Sudden, gradual and recurrent concept drift
- A supervised Random Forest NIDS baseline
- Distribution-based drift detection using a KS statistic
- Error-based DDM-style detection
- Triggered retraining on a recent labeled window
- Static vs adaptive model comparison
- Accuracy, precision, recall, F1, FPR, FNR and G-Mean
- Drift score, detection events, update time and inference time
- Interactive Streamlit dashboard

The survey reports that commonly used drift datasets include MAWI, DS2OS Traffic Traces, IP Maliciousness and UGR'16, while CIC-IDS2017, KDDCup 1999, NSL-KDD, CSE-CIC-IDS2018 and IoTID20 are among the most used static datasets. This demo uses synthetic data so it can run immediately without downloading a large cybersecurity dataset.

## Run locally

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

## Deployment

Recommended free option: Streamlit Community Cloud.

1. Create a GitHub repository.
2. Upload the project files.
3. Open Streamlit Community Cloud.
4. Select the repository and `app.py`.
5. Deploy.

No database, API key or paid service is required.

## Research limitation

This is not a reproduction of the survey's 69-paper empirical review. It is a student-built executable prototype that converts the survey's core NIDS concepts—drift generation, detection, adaptation and dynamic evaluation—into an interactive experiment.

Synthetic experiment results must not be presented as results reported by the IEEE paper.
