"""LSTM model definitions and persistence."""

from __future__ import annotations

from typing import Any

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.models import Sequential, load_model


def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_true - y_pred)))


MODEL_ALIASES = {
    "single": "single",
    "double": "double",
    "bidir": "bidirectional",
    "bidirectional": "bidirectional",
}


def build_lstm_model(model_type: str, input_shape: tuple[int, int], cfg: dict[str, Any]) -> Sequential:
    units = cfg.get("training", {}).get("lstm_units", 64)
    dropout = cfg.get("training", {}).get("dropout", 0.1)
    model_type = MODEL_ALIASES.get(model_type, model_type)

    model = Sequential()
    if model_type == "single":
        model.add(LSTM(units, input_shape=input_shape))
    elif model_type == "double":
        model.add(LSTM(units, return_sequences=True, input_shape=input_shape))
        model.add(LSTM(units))
    elif model_type == "bidirectional":
        model.add(Bidirectional(LSTM(units), input_shape=input_shape))
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.add(Dropout(dropout))
    model.add(Dense(1))
    return model


def compile_model(model: Sequential, cfg: dict[str, Any]) -> Sequential:
    opt_name = cfg.get("training", {}).get("optimizer", "adam")
    optimizer = opt_name if isinstance(opt_name, str) else "adam"
    model.compile(optimizer=optimizer, loss=rmse, metrics=["mae"])
    return model


def load_trained_model(path: str):
    return load_model(path, custom_objects={"rmse": rmse})
