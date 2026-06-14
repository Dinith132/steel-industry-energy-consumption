"""Minimal Keras LSTM forward pass in NumPy (no TensorFlow required)."""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import h5py
import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _lstm_cell(
    x: np.ndarray,
    h: np.ndarray,
    c: np.ndarray,
    kernel: np.ndarray,
    recurrent: np.ndarray,
    bias: np.ndarray,
    units: int,
) -> tuple[np.ndarray, np.ndarray]:
    z = x @ kernel + h @ recurrent + bias
    i = _sigmoid(z[:, :units])
    f = _sigmoid(z[:, units : 2 * units])
    c_g = np.tanh(z[:, 2 * units : 3 * units])
    o = _sigmoid(z[:, 3 * units :])
    c_new = f * c + i * c_g
    h_new = o * np.tanh(c_new)
    return h_new, c_new


def lstm_forward(
    x: np.ndarray,
    kernel: np.ndarray,
    recurrent: np.ndarray,
    bias: np.ndarray,
    return_sequences: bool = False,
) -> np.ndarray:
    """x: (batch, timesteps, features) -> (batch, units) or (batch, timesteps, units)."""
    batch, steps, _ = x.shape
    units = recurrent.shape[0]
    h = np.zeros((batch, units), dtype=np.float32)
    c = np.zeros((batch, units), dtype=np.float32)
    outputs = []
    for t in range(steps):
        h, c = _lstm_cell(x[:, t, :], h, c, kernel, recurrent, bias, units)
        if return_sequences:
            outputs.append(h)
    if return_sequences:
        return np.stack(outputs, axis=1)
    return h


def load_keras_zip(model_path: Path) -> dict:
    with zipfile.ZipFile(model_path, "r") as zf:
        config = json.loads(zf.read("config.json"))
        weights_bytes = zf.read("model.weights.h5")
    tmp = model_path.parent / f"._{model_path.stem}_weights.h5"
    tmp.write_bytes(weights_bytes)
    weights = h5py.File(tmp, "r")
    return {"config": config, "weights": weights, "tmp": tmp}


def _get_lstm_weights(wg, prefix: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    g = wg[f"layers/{prefix}/cell/vars"]
    kernel = np.array(g["0"])
    recurrent = np.array(g["1"])
    bias = np.array(g["2"])
    return kernel, recurrent, bias


def predict_hidden(
    model_path: Path,
    X: np.ndarray,
    layer: str = "final_lstm",
) -> np.ndarray:
    """
    layer options:
      - 'final_lstm': last LSTM output (single/double layer2/bidir concat)
      - 'first_lstm_last_step': first LSTM layer, last timestep (matches existing notebooks)
      - 'first_lstm_all_steps': (batch, timesteps, units) for double only
    """
    bundle = load_keras_zip(model_path)
    cfg = bundle["config"]
    wg = bundle["weights"]
    layers = [l for l in cfg["config"]["layers"] if l["class_name"] != "InputLayer"]

    x = X.astype(np.float32)
    out = None

    for layer_cfg in layers:
        name = layer_cfg["class_name"]
        if name == "LSTM":
            lc = layer_cfg["config"]
            rs = lc["return_sequences"]
            # find matching weight group (first lstm vs lstm_1)
            if "lstm_1" in wg["layers"]:
                if "first_lstm_last_step" in (layer, "first_lstm_all_steps") and out is None:
                    k, r, b = _get_lstm_weights(wg, "lstm")
                    seq = lstm_forward(x, k, r, b, return_sequences=True)
                    if layer == "first_lstm_all_steps":
                        bundle["weights"].close()
                        bundle["tmp"].unlink(missing_ok=True)
                        return seq
                    out = seq[:, -1, :]
                    x = seq
                    continue
                k, r, b = _get_lstm_weights(wg, "lstm_1")
                out = lstm_forward(x, k, r, b, return_sequences=rs)
                if rs:
                    x = out
            else:
                k, r, b = _get_lstm_weights(wg, "lstm")
                seq = lstm_forward(x, k, r, b, return_sequences=rs)
                if layer == "first_lstm_all_steps" and rs:
                    bundle["weights"].close()
                    bundle["tmp"].unlink(missing_ok=True)
                    return seq
                if layer == "first_lstm_last_step" and rs:
                    out = seq[:, -1, :]
                else:
                    out = seq if rs else seq
                    if rs:
                        x = seq

        elif name == "Bidirectional":
            kf, rf, bf = _get_lstm_weights(wg, "bidirectional/forward_layer")
            kb, rb, bb = _get_lstm_weights(wg, "bidirectional/backward_layer")
            hf = lstm_forward(x, kf, rf, bf, return_sequences=False)
            x_rev = x[:, ::-1, :]
            hb = lstm_forward(x_rev, kb, rb, bb, return_sequences=False)
            out = np.concatenate([hf, hb], axis=1)

        elif name in ("Dropout", "Dense"):
            continue

    bundle["weights"].close()
    bundle["tmp"].unlink(missing_ok=True)
    return out
