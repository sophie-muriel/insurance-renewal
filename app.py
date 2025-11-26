import os
import numpy as np
from flask import Flask, request, render_template, jsonify
import pickle
from huggingface_hub import hf_hub_download
import threading

app = Flask(__name__)

# global variables
model = None
scaler = None
is_ready = False
load_lock = threading.Lock()


def background_load():
    """Downloads and loads the model and scaler in the background to not block the website."""
    global model, scaler, is_ready

    if is_ready:
        return

    with load_lock:
        try:
            local_model_path = "insurance_renewal_model.pkl"
            local_scaler_path = "scaler.pkl"

            if os.path.exists(local_model_path) and os.path.exists(local_scaler_path):
                print(">>> LOADING FROM LOCAL FILES...", flush=True)

                with open(local_model_path, "rb") as f:
                    model = pickle.load(f)
                with open(local_scaler_path, "rb") as f:
                    scaler = pickle.load(f)
            else:
                print(">>> STARTING MODEL DOWNLOAD...", flush=True)

                model_path = hf_hub_download(
                    repo_id="sophie-muriel/renovacion-seguros",
                    filename="insurance_renewal_model.pkl",
                    cache_dir="."
                )
                scaler_path = hf_hub_download(
                    repo_id="sophie-muriel/renovacion-seguros",
                    filename="scaler.pkl",
                    cache_dir="."
                )

                # open with Pickle
                with open(model_path, "rb") as f:
                    model = pickle.load(f)
                with open(scaler_path, "rb") as f:
                    scaler = pickle.load(f)

            is_ready = True
            print(">>> MODEL LOADED SUCCESSFULLY.", flush=True)

        except Exception as e:
            print(f"ERROR LOADING MODEL: {str(e)}", flush=True)


# thread (background loading basically)
threading.Thread(target=background_load).start()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/status")
def status():
    return jsonify({"ready": is_ready})


@app.route("/predict", methods=["POST"])
def predict():
    if not is_ready:
        return render_template(
            "index.html",
            prediction_text="MODEL LOADING /// PLEASE WAIT."
        )

    try:
        # features in order
        form_fields = [
            "perc_premium_paid_by_cash_credit",
            "income",
            "application_underwriting_score",
            "age_in_years",
            "total_late_payments",
            "has_late_payments"
        ]

        form_values = [float(request.form[field]) for field in form_fields]

        # 'income' logarithmic transformation
        form_values[1] = np.log(form_values[1] + 1)

        full_numeric = np.array([[
            form_values[0],  # perc_premium
            form_values[1],  # income (logged)
            form_values[2],  # app_score
            0,               # no_of_premiums_paid (dummy)
            0,               # premium (dummy)
            form_values[3],  # age
            form_values[4]   # total_late
        ]])

        scaled_full = scaler.transform(full_numeric)

        # final features for the model
        selected_indices = [0, 1, 2, 5, 6]
        X_final = scaled_full[:, selected_indices]

        # binary feature (not scaled)
        has_late = form_values[5]
        X_input = np.hstack([X_final, np.array([[has_late]])])

        # prediction
        prob_renew = model.predict_proba(X_input)[0][1]

        if prob_renew >= 0.5:
            result_text = f"PROBABLE RENEWAL // PROBABILITY: <span class='prob-accent'>{prob_renew*100:.1f}%</span>"
        else:
            result_text = f"IMPROBABLE RENEWAL (CHURN) // PROBABILITY: <span class='prob-accent'>{prob_renew*100:.1f}%</span>"

        return render_template("index.html", prediction_text=result_text)
    except Exception as e:
        return f"ERROR: {str(e)}", 500


@app.route("/health")
def health():
    return "ok", 200


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
