# app.py
import os
import json
import datetime

from flask import (
    Flask, render_template, request, redirect, url_for, flash
)
from werkzeug.utils import secure_filename

import pandas as pd
import numpy as np
import tensorflow as tf

# --- CONFIGURATION ---
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"txt", "csv"}

app = Flask(__name__)
app.secret_key = "replace-with-a-secret-key"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

MODEL_PATH       = "models/final_model.h5"
WINDOW_SIZE      = 24
FORECAST_HORIZON = 1

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
model = tf.keras.models.load_model(MODEL_PATH, compile=False)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def make_comparison_df_from_file(f):
    df = pd.read_csv(
        f,
        sep=";",
        decimal=".",
        usecols=["Global_active_power"],
        na_values="?"
    ).dropna()

    series = df["Global_active_power"].astype(np.float32).values
    recs = []
    for i in range(len(series) - WINDOW_SIZE - FORECAST_HORIZON + 1):
        window   = series[i : i + WINDOW_SIZE]
        true_val = float(series[i + WINDOW_SIZE])
        pred_val = float(model.predict(window.reshape(1, -1), verbose=0)[0, 0])
        recs.append((true_val, pred_val))

    return pd.DataFrame(recs, columns=["valeurs_reelles", "valeurs_predites"])


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Upload handling
        if "file" not in request.files:
            flash("Aucun fichier sélectionné", "danger")
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            flash("Aucun fichier sélectionné", "danger")
            return redirect(request.url)

        if not allowed_file(file.filename):
            flash("Extension non autorisée (txt, csv seulement)", "danger")
            return redirect(request.url)

        # Build comparison DF
        df_cmp = make_comparison_df_from_file(file)

        # Prepare data for Chart.js
        labels    = list(range(len(df_cmp)))
        true_vals = df_cmp["valeurs_reelles"].tolist()
        pred_vals = df_cmp["valeurs_predites"].tolist()

        # Errors & metrics
        diff    = np.array(pred_vals) - np.array(true_vals)
        total   = len(df_cmp)
        rmse    = float(np.sqrt((diff**2).mean()))
        mae     = float(np.abs(diff).mean())
        mape    = float((np.abs(diff / np.array(true_vals))).mean() * 100)
        err_std = float(diff.std())

        # Build HTML table
        table_html = df_cmp.to_html(
            classes="table table-striped table-hover",
            index=False
        )

        return render_template(
            "index.html",
            now=datetime.datetime.now(),
            total=total,
            labels=json.dumps(labels),
            true_vals=json.dumps(true_vals),
            pred_vals=json.dumps(pred_vals),
            rmse=rmse,
            mae=mae,
            mape=mape,
            err_std=err_std,
            table_html=table_html
        )

    # GET → only upload form
    return render_template("index.html", now=datetime.datetime.now())


if __name__ == "__main__":
    app.run(debug=True)