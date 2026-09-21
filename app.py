import streamlit as st
import pandas as pd
import plotly.express as px
from src.pipeline import run_experiment

st.set_page_config(page_title="Concept-Drift-Resilient NIDS", page_icon="🛡️", layout="wide")

st.title("🛡️ Concept-Drift-Resilient Network Intrusion Detection System")
st.caption("Research-oriented simulator based on the 2026 IEEE Access survey on concept-drift-resilient NIDS.")

with st.sidebar:
    st.header("Experiment")
    samples = st.slider("Samples", 1000, 8000, 3000, 500)
    features = st.slider("Features", 8, 30, 12)
    drift_type = st.selectbox("Drift type", ["Sudden", "Gradual", "Recurrent"])
    drift_point = st.slider("Drift point (%)", 30, 70, 50)
    detector = st.selectbox("Drift detector", ["KS Distribution Test", "DDM Error Detector"])
    adaptation = st.selectbox("Adaptation", ["Triggered Retraining", "Static Model"])
    window = st.slider("Recent-data window", 100, 1000, 400, 100)
    seed = st.number_input("Random seed", 1, 9999, 42)
    run = st.button("Run experiment", type="primary")

st.info("Offline educational simulator. It does not capture, modify, or attack live network traffic.")

if run or "result" not in st.session_state:
    with st.spinner("Generating drifting traffic stream and evaluating NIDS..."):
        st.session_state.result = run_experiment(
            n_samples=samples, n_features=features, drift_type=drift_type,
            drift_point_pct=drift_point, detector=detector,
            adaptation=adaptation, recent_window=window, seed=int(seed)
        )

r = st.session_state.result

m = st.columns(6)
m[0].metric("Static F1", f"{r['static']['f1']:.3f}")
m[1].metric("Adaptive F1", f"{r['adaptive']['f1']:.3f}")
m[2].metric("Static FPR", f"{r['static']['fpr']:.3f}")
m[3].metric("Adaptive FPR", f"{r['adaptive']['fpr']:.3f}")
m[4].metric("Drift events", str(r["drift_events"]))
m[5].metric("Adaptations", str(r["adaptations"]))

tabs = st.tabs(["Overview", "Stream Performance", "Drift Detection", "Adaptation", "Case Study"])

with tabs[0]:
    st.subheader("Why concept drift matters")
    st.write(
        "Network traffic is non-stationary: the relationship between traffic features and labels "
        "can change over time. A model trained on an old distribution can therefore degrade after deployment."
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Experiment design**")
        st.write(f"- Drift: {drift_type}")
        st.write(f"- Drift point: approximately {drift_point}% of stream")
        st.write(f"- Detector: {detector}")
        st.write(f"- Adaptation: {adaptation}")
    with c2:
        st.markdown("**Paper-aligned concepts**")
        st.write("- Sudden, gradual and recurrent drift")
        st.write("- Distribution/error-based drift detection")
        st.write("- Triggered retraining")
        st.write("- Dynamic evaluation beyond accuracy")

with tabs[1]:
    df = pd.DataFrame(r["timeline"])
    fig = px.line(df, x="chunk", y=["static_f1", "adaptive_f1"], markers=True,
                  labels={"value":"F1-score", "variable":"Model"})
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df, use_container_width=True)

with tabs[2]:
    df = pd.DataFrame(r["timeline"])
    fig = px.line(df, x="chunk", y="drift_score", markers=True,
                  labels={"drift_score":"Drift score"})
    fig.add_hline(y=r["threshold"], line_dash="dash", annotation_text="threshold")
    st.plotly_chart(fig, use_container_width=True)
    st.write(f"Detector threshold: **{r['threshold']:.3f}**")
    st.write("Detected drift chunks:", r["detected_chunks"] or "None")

with tabs[3]:
    st.subheader("Triggered adaptation")
    st.write(
        "When drift is detected, the simulator retrains the classifier on a recent labeled window. "
        "This represents the survey's triggered-retraining family of adaptation strategies."
    )
    st.write(f"Number of retraining events: **{r['adaptations']}**")
    st.write(f"Total retraining/update time: **{r['update_time']:.3f} s**")
    st.write(f"Average inference time/sample: **{r['inference_time']*1000:.3f} ms**")

with tabs[4]:
    st.subheader("One drift event")
    st.write(r["case_description"])
    st.dataframe(pd.DataFrame(r["case_rows"]), use_container_width=True)

st.divider()
st.caption(
    "Base paper: M. Komarchesqui et al., “A Comprehensive Survey on Concept-Drift-Resilient "
    "Network Intrusion Detection Systems,” IEEE Access, 2026, DOI: 10.1109/ACCESS.2026.3691262. "
    "This implementation is a student research simulator inspired by the survey; its synthetic "
    "results are not claimed as results reported by the paper."
)
