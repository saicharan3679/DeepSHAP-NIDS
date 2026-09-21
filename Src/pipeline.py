import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import ks_2samp

def make_stream(n=3000, d=12, drift_type="Sudden", drift_point_pct=50, seed=42):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, (n, d))
    y = (x[:, 0] + 0.7*x[:, 1] + 0.4*x[:, 2] + rng.normal(0, 0.8, n) > 0).astype(int)
    point = int(n * drift_point_pct / 100)

    if drift_type == "Sudden":
        idx = np.arange(point, n)
        x[idx, 0] += 1.4
        x[idx, 3] -= 1.0
        y[idx] = (x[idx, 0] - 0.9*x[idx, 1] + 0.7*x[idx, 4] + rng.normal(0, 0.8, len(idx)) > 1.0).astype(int)
    elif drift_type == "Gradual":
        start, end = max(0, point-400), min(n, point+400)
        for i in range(start, end):
            a = (i-start) / max(1, end-start)
            x[i, 0] = (1-a)*x[i, 0] + a*(x[i, 0] + 1.5)
            x[i, 3] = (1-a)*x[i, 3] + a*(x[i, 3] - 1.2)
            if a > 0.5:
                y[i] = (x[i, 0] - 0.9*x[i, 1] + 0.7*x[i, 4] + rng.normal(0, 0.8) > 1.0).astype(int)
    else:
        period = 500
        for start in range(point, n, period):
            idx = np.arange(start, min(n, start+period))
            phase = ((start-point)//period) % 2
            if phase == 0:
                x[idx, 0] += 1.4
                y[idx] = (x[idx,0] - 0.9*x[idx,1] + 0.7*x[idx,4] + rng.normal(0,0.8,len(idx)) > 1.0).astype(int)
            else:
                x[idx, 0] -= 1.0
                y[idx] = (x[idx,0] + 0.7*x[idx,1] + 0.4*x[idx,2] + rng.normal(0,0.8,len(idx)) > 0).astype(int)

    return x, y, point

def metrics(y, p):
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0,1]).ravel()
    return {
        "accuracy": accuracy_score(y,p),
        "precision": precision_score(y,p,zero_division=0),
        "recall": recall_score(y,p,zero_division=0),
        "f1": f1_score(y,p,zero_division=0),
        "fpr": fp/max(1,fp+tn),
        "fnr": fn/max(1,fn+tp),
        "gmean": np.sqrt((tn/max(1,tn+fp))*(tp/max(1,tp+fn))),
    }

def detect_ks(reference, recent):
    scores = []
    for j in range(reference.shape[1]):
        scores.append(ks_2samp(reference[:,j], recent[:,j]).statistic)
    return float(np.mean(scores))

def detect_ddm(errors):
    n = len(errors)
    if n < 30:
        return 0.0
    p = np.mean(errors)
    s = np.sqrt(max(1e-12, p*(1-p)/n))
    return float(p + 2*s)

def run_experiment(n_samples=3000, n_features=12, drift_type="Sudden",
                   drift_point_pct=50, detector="KS Distribution Test",
                   adaptation="Triggered Retraining", recent_window=400, seed=42):
    X, y, point = make_stream(n_samples, n_features, drift_type, drift_point_pct, seed)
    scaler = StandardScaler()
    X0 = scaler.fit_transform(X[:point])
    Xs = scaler.transform(X)

    split = max(200, int(point*0.75))
    X_train, X_val = Xs[:split], Xs[split:point]
    y_train, y_val = y[:split], y[split:point]

    model_static = RandomForestClassifier(n_estimators=80, random_state=seed, n_jobs=-1)
    model_static.fit(X_train, y_train)
    model_adapt = RandomForestClassifier(n_estimators=80, random_state=seed, n_jobs=-1)
    model_adapt.fit(X_train, y_train)

    chunk = max(50, min(200, n_samples//15))
    rows = []
    detected = []
    adaptations = 0
    update_time = 0.0
    ref = Xs[max(0, split-500):split]
    error_history = []

    for start in range(point, n_samples, chunk):
        end = min(n_samples, start+chunk)
        xb, yb = Xs[start:end], y[start:end]

        ps = model_static.predict(xb)
        pa = model_adapt.predict(xb)
        ms = metrics(yb, ps)
        ma = metrics(yb, pa)

        if detector.startswith("KS"):
            score = detect_ks(ref, xb)
            threshold = 0.18
        else:
            errs = (pa != yb).astype(int).tolist()
            error_history.extend(errs)
            score = detect_ddm(error_history[-300:])
            threshold = 0.42

        drift = score > threshold
        if drift:
            detected.append(start//chunk)
            if adaptation == "Triggered Retraining":
                t0 = time.perf_counter()
                lo = max(0, start-recent_window)
                model_adapt = RandomForestClassifier(n_estimators=80, random_state=seed+adaptations+1, n_jobs=-1)
                model_adapt.fit(Xs[lo:start], y[lo:start])
                update_time += time.perf_counter()-t0
                adaptations += 1

        rows.append({
            "chunk": start//chunk,
            "start": start,
            "static_f1": ms["f1"],
            "adaptive_f1": ma["f1"],
            "static_accuracy": ms["accuracy"],
            "adaptive_accuracy": ma["accuracy"],
            "static_fpr": ms["fpr"],
            "adaptive_fpr": ma["fpr"],
            "drift_score": score,
            "drift": drift
        })
        ref = np.vstack([ref, xb])[-500:]

    X_test = Xs[point:]
    y_test = y[point:]
    ps = model_static.predict(X_test)
    pa = model_adapt.predict(X_test)
    sm, am = metrics(y_test, ps), metrics(y_test, pa)

    t0 = time.perf_counter()
    _ = model_adapt.predict(X_test[:min(200,len(X_test))])
    inference_time = (time.perf_counter()-t0)/max(1,min(200,len(X_test)))

    case_start = max(point, point+chunk*2)
    case_end = min(n_samples, case_start+min(50, chunk))
    case_p = model_adapt.predict(Xs[case_start:case_end])
    case_rows = [{
        "sample": i,
        "actual": int(y[i]),
        "prediction": int(case_p[i-case_start]),
        "correct": bool(y[i] == case_p[i-case_start])
    } for i in range(case_start, case_end)]

    return {
        "static": sm, "adaptive": am, "timeline": rows,
        "drift_events": len(detected), "detected_chunks": detected,
        "adaptations": adaptations, "update_time": update_time,
        "inference_time": inference_time, "threshold": threshold,
        "case_rows": case_rows,
        "case_description": f"A {drift_type.lower()} drift stream was generated around sample {point}. "
                            "The table shows a small post-drift prediction window."
                        }
