# DeepSHAP-NIDS

A student research prototype demonstrating:

1. Synthetic network-flow input
2. Neural-network NIDS prediction
3. DeepSHAP explanation of the prediction
4. A small gradient-based adversarial robustness test
5. Before/after comparison

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Important

This is an educational offline simulator. It does not capture, modify, or attack live network traffic.

The current repository version was a concept-drift Random Forest simulator. This version changes the application workflow to match the stated DeepSHAP-NIDS project direction.
