
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import shap
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

st.set_page_config(
    page_title="DeepSHAP-NIDS | Adversarial Intrusion Detection",
    page_icon="🛡️",
    layout="wide",
)

FEATURE_NAMES = [
    "Flow Duration", "Packet Count", "Avg Packet Size", "Bytes/Second",
    "Packets/Second", "Forward Packets", "Backward Packets",
    "TCP Flag Count", "Active Time", "Idle Time"
]

@st.cache_data
def make_dataset(n=3000, seed=42):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, (n, len(FEATURE_NAMES)))
    x[:, 0] = np.abs(x[:, 0]) * 1.2
    x[:, 1] = np.abs(x[:, 1]) * 1.5
    x[:, 2] = np.abs(x[:, 2]) * 0.9
    score = (
        1.2*x[:, 1] + 0.9*x[:, 3] + 0.8*x[:, 7]
        - 0.7*x[:, 4] + 0.5*x[:, 6] + rng.normal(0, 1.0, n)
    )
    y = (score > 0.9).astype(np.float32)
    return x.astype(np.float32), y

class NIDSModel(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(1)

@st.cache_resource
def train_model(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    X, y = make_dataset(seed=seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_test_s = scaler.transform(X_test).astype(np.float32)

    model = NIDSModel(X_train_s.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.BCEWithLogitsLoss()
    Xt = torch.tensor(X_train_s)
    yt = torch.tensor(y_train)

    model.train()
    for _ in range(80):
        opt.zero_grad()
        loss = loss_fn(model(Xt), yt)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        p = (torch.sigmoid(model(torch.tensor(X_test_s))) >= 0.5).numpy().astype(int)
    return model, scaler, X_test_s, y_test.astype(int), p

def predict(model, x):
    model.eval()
    with torch.no_grad():
        prob = torch.sigmoid(model(torch.tensor(x, dtype=torch.float32))).numpy()
    return prob

def metrics(y, p):
    return {
        "Accuracy": accuracy_score(y, p),
        "Precision": precision_score(y, p, zero_division=0),
        "Recall": recall_score(y, p, zero_division=0),
        "F1": f1_score(y, p, zero_division=0),
    }

def explain(model, background, sample):
    # DeepSHAP / SHAP DeepExplainer for the neural NIDS model.
    explainer = shap.DeepExplainer(model, torch.tensor(background[:100], dtype=torch.float32))
    values = explainer.shap_values(torch.tensor(sample, dtype=torch.float32))
    if isinstance(values, list):
        values = values[0]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, 0]
    return values

def fgsm_attack(model, x, epsilon):
    x_t = torch.tensor(x.copy(), dtype=torch.float32, requires_grad=True)
    model.zero_grad()
    score = model(x_t).sum()
    score.backward()
    # For a binary intrusion score, moving opposite to the gradient
    # attempts to reduce the model's attack score.
    adv = x_t - epsilon * x_t.grad.sign()
    return adv.detach().numpy()

# ---------- Header ----------
st.markdown(
    """
    <style>
    .hero {padding: 18px 22px; border-radius: 18px; background: linear-gradient(135deg,#101827,#17243a);}
    .hero h1 {margin:0;color:white;font-size:2.1rem;}
    .hero p {color:#c8d3e3;margin:8px 0 0;}
    .step {padding:12px;border:1px solid #d8dee8;border-radius:12px;text-align:center;}
    .small {font-size:0.88rem;color:#667085;}
    </style>
    <div class="hero">
      <h1>🛡️ DeepSHAP-NIDS</h1>
      <p>Explainable Network Intrusion Detection with an adversarial robustness demonstration</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")
steps = st.columns(5)
for c, n, label in zip(
    steps,
    ["1", "2", "3", "4", "5"],
    ["Network Input", "NIDS Prediction", "DeepSHAP", "Adversarial Test", "Before vs After"]
):
    c.markdown(f'<div class="step"><b>{n}</b><br>{label}</div>', unsafe_allow_html=True)

st.info(
    "This is an offline educational prototype. It uses synthetic network-flow features; "
    "it does not capture, modify, or attack live network traffic."
)

with st.sidebar:
    st.header("Experiment Controls")
    seed = st.number_input("Random seed", 1, 9999, 42)
    epsilon = st.slider(
        "Adversarial perturbation strength",
        0.00, 0.50, 0.08, 0.01,
        help="Small demonstration perturbation applied to the selected test sample."
    )
    sample_id = st.slider("Test sample", 0, 99, 0)
    run_attack = st.button("Run complete experiment", type="primary")

model, scaler, X_test, y_test, baseline_pred = train_model(int(seed))

if run_attack or "adv_result" not in st.session_state:
    x = X_test[sample_id:sample_id+1]
    before_prob = float(predict(model, x)[0])
    adv_x = fgsm_attack(model, x, epsilon)
    after_prob = float(predict(model, adv_x)[0])

    before_label = int(before_prob >= 0.5)
    after_label = int(after_prob >= 0.5)

    st.session_state.adv_result = {
        "x": x, "adv_x": adv_x,
        "before_prob": before_prob, "after_prob": after_prob,
        "before_label": before_label, "after_label": after_label
    }

r = st.session_state.adv_result

# ---------- Step 1 ----------
st.header("1. Network Traffic Input")
input_df = pd.DataFrame({
    "Feature": FEATURE_NAMES,
    "Value": r["x"][0],
    "After adversarial perturbation": r["adv_x"][0],
})
st.dataframe(input_df, use_container_width=True, hide_index=True)

# ---------- Step 2 ----------
st.header("2. NIDS Prediction")
a, b, c = st.columns(3)
a.metric("Actual class", "Attack" if y_test[sample_id] else "Normal")
b.metric("Model prediction", "Attack" if r["before_label"] else "Normal")
c.metric("Attack probability", f"{r['before_prob']*100:.1f}%")

# ---------- Step 3 ----------
st.header("3. DeepSHAP Explanation")
st.write(
    "DeepSHAP estimates how each input feature contributes to the neural NIDS prediction. "
    "Larger absolute SHAP values indicate a stronger contribution for this sample."
)

with st.spinner("Computing DeepSHAP explanation..."):
    try:
        shap_values = explain(model, X_test, r["x"])
        sv = shap_values[0]
        shap_df = pd.DataFrame({"Feature": FEATURE_NAMES, "SHAP value": sv})
        shap_df["Absolute impact"] = shap_df["SHAP value"].abs()
        shap_df = shap_df.sort_values("Absolute impact", ascending=True)
        fig = px.bar(
            shap_df, x="SHAP value", y="Feature", orientation="h",
            title="DeepSHAP feature contribution"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(
            shap_df.sort_values("Absolute impact", ascending=False)
                    .drop(columns="Absolute impact"),
            use_container_width=True, hide_index=True
        )
    except Exception as e:
        st.warning(
            "DeepSHAP could not be rendered in this runtime. "
            "The prediction and adversarial comparison remain available."
        )
        st.caption(f"Technical detail: {type(e).__name__}: {e}")

# ---------- Step 4 ----------
st.header("4. Adversarial Robustness Test")
st.write(
    "The demonstration applies a small gradient-based perturbation to the selected "
    "synthetic flow. The goal is to test whether the model's classification is sensitive "
    "to small input changes."
)
d1, d2, d3 = st.columns(3)
d1.metric("Perturbation strength", f"{epsilon:.2f}")
d2.metric("Before attack", "Attack" if r["before_label"] else "Normal")
d3.metric("After attack", "Attack" if r["after_label"] else "Normal")

# ---------- Step 5 ----------
st.header("5. Before vs After")
compare = pd.DataFrame({
    "Stage": ["Original", "Adversarial"],
    "Attack probability": [r["before_prob"], r["after_prob"]],
})
fig2 = px.bar(compare, x="Stage", y="Attack probability", range_y=[0,1],
              text_auto=".1%", title="Model confidence comparison")
st.plotly_chart(fig2, use_container_width=True)

if r["before_label"] != r["after_label"]:
    st.success(
        "The selected perturbation changed the model's predicted class. "
        "This is an example of adversarial sensitivity in the demonstration model."
    )
else:
    st.warning(
        "The predicted class did not change for this sample and perturbation strength. "
        "Try another test sample or a slightly larger demonstration perturbation."
    )

# ---------- Overall model ----------
st.divider()
st.subheader("Model-level evaluation")
base_m = metrics(y_test, baseline_pred)
cols = st.columns(4)
for c, (k, v) in zip(cols, base_m.items()):
    c.metric(k, f"{v:.3f}")

st.caption(
    "Research note: this prototype is a student-built demonstration. "
    "Synthetic results are not results reported by the source research paper, "
    "and the adversarial experiment should not be interpreted as an evaluation "
    "of a production NIDS."
)
