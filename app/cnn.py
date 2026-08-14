import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "model"
CNN_WEIGHTS_JSON = MODEL_DIR / "cnn_weights.json"

V3_TARGETS = ["rain120_15mm", "rain120_25mm", "rain120_35mm"]
V3_LABELS = {
    "rain120_15mm": "Rain ≥15mm/30min in ~2h (CNN)",
    "rain120_25mm": "Rain ≥25mm/30min in ~2h (CNN)",
    "rain120_35mm": "Rain ≥35mm/30min in ~2h (CNN)",
}
THRESHOLDS_MM = [15.0, 25.0, 35.0]

IN_CH = 5  # v4 (P7): 4 lead frames + delta(lead1-lead0); old models used 3
SIZE = 32
F1, F2, HID = 8, 16, 32
INPUT_SCALE = 50.0


def relu(x):
    return np.maximum(x, 0.0)


def average_precision(y, s):
    y = np.asarray(y, dtype=float)
    s = np.asarray(s, dtype=float)
    n_pos = y.sum()
    if n_pos == 0:
        return 0.0
    order = np.argsort(-s, kind="stable")
    y = y[order]
    tp = np.cumsum(y)
    prec = tp / (np.arange(len(y)) + 1)
    return float((prec * y).sum() / n_pos)


def brier(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(np.mean((p - y) ** 2))


def sigmoid(x):
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


def im2col(X, kh=3, kw=3):
    N, C, H, W = X.shape
    Hp, Wp = H - kh + 1, W - kw + 1
    cols = np.zeros((N, C, kh, kw, Hp, Wp), dtype=X.dtype)
    for i in range(kh):
        for j in range(kw):
            cols[:, :, i, j, :, :] = X[:, :, i:i + Hp, j:j + Wp]
    return cols.transpose(0, 4, 5, 1, 2, 3).reshape(N, Hp, Wp, C * kh * kw)


def col2im(cols, N, C, H, W, kh=3, kw=3):
    Hp, Wp = H - kh + 1, W - kw + 1
    out = np.zeros((N, C, H, W), dtype=cols.dtype)
    cols_r = cols.reshape(N, Hp, Wp, C, kh, kw)
    for i in range(kh):
        for j in range(kw):
            out[:, :, i:i + Hp, j:j + Wp] += cols_r[:, :, :, :, i, j].transpose(0, 3, 1, 2)
    return out


def conv3x3(X, W, b):
    N, C, H, Wd = X.shape
    F = W.shape[0]
    Xp = np.pad(X, ((0, 0), (0, 0), (1, 1), (1, 1)))
    cols = im2col(Xp)
    out = cols @ W.reshape(F, -1).T + b.reshape(1, 1, 1, F)
    return out.transpose(0, 3, 1, 2)


def conv3x3_back(dout, X, W):
    N, C, H, Wd = X.shape
    F = W.shape[0]
    Xp = np.pad(X, ((0, 0), (0, 0), (1, 1), (1, 1)))
    cols = im2col(Xp)
    dout_perm = dout.transpose(0, 2, 3, 1)
    dW = np.einsum("nhwq,nhwf->fq", cols, dout_perm).reshape(W.shape)
    db = dout.sum(axis=(0, 2, 3))
    dcols = dout_perm @ W.reshape(F, -1)
    dXp = col2im(dcols, N, C, H + 2, Wd + 2)
    return dXp[:, :, 1:-1, 1:-1], dW, db


def maxpool2x2(X):
    N, C, H, W = X.shape
    Hp, Wp = H // 2, W // 2
    Xr = X.reshape(N, C, Hp, 2, Wp, 2)
    return Xr.max(axis=(3, 5))


def maxpool2x2_back(dout, X):
    N, C, H, W = X.shape
    Hp, Wp = H // 2, W // 2
    Xr = X.reshape(N, C, Hp, 2, Wp, 2)
    mask = Xr == Xr.max(axis=(3, 5), keepdims=True)
    dXr = np.zeros_like(Xr)
    dout_rep = dout[:, :, :, None, :, None]
    dXr[mask] = np.broadcast_to(dout_rep, Xr.shape)[mask]
    return dXr.reshape(N, C, H, W)


def init_weights(seed=0, in_ch=IN_CH):
    rng = np.random.default_rng(seed)
    return [
        rng.normal(0, 0.15, (F1, in_ch, 3, 3)).astype(np.float32),
        np.zeros(F1, dtype=np.float32),
        rng.normal(0, 0.15, (F2, F1, 3, 3)).astype(np.float32),
        np.zeros(F2, dtype=np.float32),
        rng.normal(0, 0.1, (F2 * 8 * 8, HID)).astype(np.float32),
        np.zeros(HID, dtype=np.float32),
        rng.normal(0, 0.1, (HID, len(V3_TARGETS))).astype(np.float32),
        np.zeros(len(V3_TARGETS), dtype=np.float32),
    ]


def forward(X, weights):
    W1, b1, W2, b2, W3, b3, W4, b4 = weights
    z = maxpool2x2(relu(conv3x3(X, W1, b1)))
    z = maxpool2x2(relu(conv3x3(z, W2, b2)))
    N = X.shape[0]
    z = z.reshape(N, -1)
    z = relu(z @ W3 + b3)
    return sigmoid(z @ W4 + b4)


def forward_steps(X, weights):
    W1, b1, W2, b2, W3, b3, W4, b4 = weights
    a1_pre = relu(conv3x3(X, W1, b1))
    a1 = maxpool2x2(a1_pre)
    a2_pre = relu(conv3x3(a1, W2, b2))
    a2 = maxpool2x2(a2_pre)
    N = X.shape[0]
    h = a2.reshape(N, -1)
    a3 = relu(h @ W3 + b3)
    return a1_pre, a1, a2_pre, a2, h, a3


def predict(X, weights):
    return forward(X, weights)


def train(X, y, B=None, epochs=40, batch=32, lr=0.05, momentum=0.9, seed=0, val_frac=0.2, quiet=False):
    """Time-ordered split (data is time-sorted; last val_frac is validation).

    B: per-sample advection/persistence baseline (max mm of the current ~2h lead
    frame). The CNN only counts as skillful on a target if it beats this baseline
    on PR-AUC and Brier (P7).
    """
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    n = X.shape[0]
    n_val = max(1, int(n * val_frac))
    Xtr, Xva = X[:n - n_val], X[n - n_val:]
    ytr, yva = y[:n - n_val], y[n - n_val:]
    Bva = np.asarray(B[n - n_val:], dtype=np.float32) if B is not None else None

    weights = init_weights(seed, in_ch=X.shape[1])
    vels = [np.zeros_like(w) for w in weights]
    best = (None, float("inf"))

    for epoch in range(epochs):
        perm = np.random.default_rng(epoch).permutation(Xtr.shape[0])
        total_loss = 0.0
        n_batches = 0
        for s in range(0, Xtr.shape[0], batch):
            idx = perm[s:s + batch]
            xb, yb = Xtr[idx], ytr[idx]
            a1_pre, a1, a2_pre, a2, h, a3 = forward_steps(xb, weights)
            W1, b1, W2, b2, W3, b3, W4, b4 = weights
            z4 = a3 @ W4 + b4
            p = sigmoid(z4)
            d4 = (p - yb) / yb.shape[0]
            dW4 = a3.T @ d4
            db4 = d4.sum(axis=0)
            da3 = d4 @ W4.T
            da3 = da3 * (a3 > 0)
            dW3 = h.T @ da3
            db3 = da3.sum(axis=0)
            dh = da3 @ W3.T
            da2 = dh.reshape(xb.shape[0], F2, 8, 8)
            da2 = maxpool2x2_back(da2, a2_pre) * (a2_pre > 0)
            da1, dW2, db2 = conv3x3_back(da2, a1, W2)
            da1 = maxpool2x2_back(da1, a1_pre) * (a1_pre > 0)
            _, dW1, db1 = conv3x3_back(da1, xb, W1)
            grads = [dW1, db1, dW2, db2, dW3, db3, dW4, db4]
            for k in range(len(weights)):
                vels[k] = momentum * vels[k] - lr * grads[k]
                weights[k] = weights[k] + vels[k]
            loss = -np.mean(yb * np.log(np.clip(p, 1e-7, 1)) + (1 - yb) * np.log(np.clip(1 - p, 1e-7, 1)))
            total_loss += loss
            n_batches += 1
        val_p = forward(Xva, weights)
        val_loss = -np.mean(yva * np.log(np.clip(val_p, 1e-7, 1)) + (1 - yva) * np.log(np.clip(1 - val_p, 1e-7, 1)))
        if not quiet:
            print(f"[cnn] epoch {epoch + 1}/{epochs} train_loss={total_loss / n_batches:.4f} val_loss={val_loss:.4f}")
        if val_loss < best[1]:
            best = ([w.copy() for w in weights], float(val_loss))

    weights = best[0]
    pva = forward(Xva, weights)
    yhat = (pva > 0.5).astype(int)
    metrics = {}
    for k, t in enumerate(V3_TARGETS):
        tp = int(((yhat[:, k] == 1) & (yva[:, k] == 1)).sum())
        fp = int(((yhat[:, k] == 1) & (yva[:, k] == 0)).sum())
        fn = int(((yhat[:, k] == 0) & (yva[:, k] == 1)).sum())
        tn = int(((yhat[:, k] == 0) & (yva[:, k] == 0)).sum())
        acc = (tp + tn) / max(1, tp + tn + fp + fn)
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        m = {
            "val_acc": round(acc, 3),
            "val_precision": round(prec, 3),
            "val_recall": round(rec, 3),
            "n_pos": int(yva[:, k].sum()),
            "pr_auc": round(average_precision(yva[:, k], pva[:, k]), 4),
            "brier": round(brier(yva[:, k], pva[:, k]), 4),
        }
        if Bva is not None:
            base_p = (Bva >= THRESHOLDS_MM[k]).astype(np.float32)
            m["baseline_pr_auc"] = round(average_precision(yva[:, k], base_p), 4)
            m["baseline_brier"] = round(brier(yva[:, k], base_p), 4)
            m["beats_baseline"] = bool(
                m["pr_auc"] >= m["baseline_pr_auc"] and m["brier"] <= m["baseline_brier"]
                and (m["pr_auc"] > m["baseline_pr_auc"] or m["brier"] < m["baseline_brier"])
            )
        metrics[t] = m
        base_note = ""
        if "beats_baseline" in m:
            base_note = (f" | baseline PR-AUC={m['baseline_pr_auc']} Brier={m['baseline_brier']} "
                         f"-> {'BEATS' if m['beats_baseline'] else 'does NOT beat'} advection baseline")
        print(f"[cnn] {t}: PR-AUC={m['pr_auc']} Brier={m['brier']} acc={acc:.3f} prec={prec:.3f} "
              f"rec={rec:.3f} (n_val={n_val}, pos={m['n_pos']}){base_note}")
    return weights, metrics


def save_weights(weights, metrics, n_total):
    MODEL_DIR.mkdir(exist_ok=True)
    payload = {
        "meta": {
            "schema_version": 4,
            "n_total": n_total,
            "generated": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
            "in_ch": int(weights[0].shape[1]),
            "input_scale": INPUT_SCALE,
        },
        "v3_targets": V3_TARGETS,
        "thresholds_mm": THRESHOLDS_MM,
        "input_scale": INPUT_SCALE,
        "w1": [w.tolist() for w in weights],
    }
    for t, m in metrics.items():
        payload[t] = m
    with open(CNN_WEIGHTS_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def load_weights():
    if not CNN_WEIGHTS_JSON.exists():
        return None
    with open(CNN_WEIGHTS_JSON, encoding="utf-8") as f:
        payload = json.load(f)
    return payload


def weights_to_list(payload):
    return [np.array(w, dtype=np.float32) for w in payload["w1"]]


def predict_frames(leads, payload=None):
    """Predict from a live snapshot's lead frames.

    Builds the channel stack the loaded model expects (meta.in_ch: 5 = 4 leads +
    delta, 3 = legacy leads[:3]). Targets whose training metrics show the CNN did
    not beat the advection baseline are dropped (P7).
    """
    if payload is None:
        payload = load_weights()
    if not payload:
        return None
    meta = payload.get("meta", {})
    if meta.get("schema_version") != 4:
        print("[cnn] skipped legacy artifact without schema_version=4")
        return None
    in_ch = int(meta.get("in_ch", 3))
    scale = payload.get("meta", {}).get("input_scale", payload.get("input_scale", INPUT_SCALE))
    if in_ch == 5:
        if len(leads) < 4:
            return None
        frames = [np.array(g, dtype=np.float32) for g in leads[:4]]
        delta = frames[1] - frames[0]
        x = np.stack(frames + [delta])[None] / scale
        x[:, :4] = np.clip(x[:, :4], 0.0, 1.0)
        x[:, 4] = np.clip(x[:, 4], -1.0, 1.0)
    else:
        if len(leads) < in_ch:
            return None
        x = np.zeros((1, in_ch, SIZE, SIZE), dtype=np.float32)
        for i in range(in_ch):
            x[0, i] = np.array(leads[i], dtype=np.float32) / scale
        x = np.clip(x, 0, 1)
    p = forward(x.astype(np.float32), weights_to_list(payload))[0]
    out = {}
    for k, t in enumerate(V3_TARGETS):
        if payload.get(t, {}).get("beats_baseline") is not True:
            continue
        out[t] = round(float(p[k]), 3)
    return out or None


def main():
    npz_path = ROOT / "data" / "grid_dataset.npz"
    if not npz_path.exists():
        print("[cnn] no dataset; run app/backfill.py first")
        return
    d = np.load(npz_path)
    X, y = d["X"], d["y"]
    B = d["B"] if "B" in d else None
    if X.shape[1] != IN_CH or B is None:
        print(
            f"[cnn] skipped: dataset schema is incompatible (channels={X.shape[1]}, "
            f"baseline={'present' if B is not None else 'missing'}); rebuild with app/backfill.py"
        )
        return
    print(f"[cnn] dataset: {X.shape[0]} samples, {X.shape[1]} channels")
    weights, metrics = train(X, y, B=B)
    save_weights(weights, metrics, int(X.shape[0]))
    print("[cnn] saved model/cnn_weights.json")


if __name__ == "__main__":
    main()
