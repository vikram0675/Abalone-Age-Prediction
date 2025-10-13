from flask import Flask, render_template, request
import pickle
import os
import numpy as np

app = Flask(__name__)

# Path to model
ROOT = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(ROOT, "abalone.pkl")

# Global variable to store last predicted age
last_predicted_age = None

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not found. Please run training/train_and_save.py to create abalone.pkl")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model

@app.route('/')
def home():
    global last_predicted_age
    return render_template('home.html', last_age=last_predicted_age)

@app.route('/predict')
def predict_page():
    return render_template('predict.html', last_age=last_predicted_age)

@app.route('/output', methods=['POST'])
def output():
    global last_predicted_age
    results = []  # store both rings and ages

    try:
        model = load_model()
        all_vals = []

        num_rows = len(request.form.getlist('sex[]'))

        for i in range(num_rows):
            vals = []
            for k in ['sex[]', 'length[]', 'diameter[]', 'height[]', 'whole[]', 'shucked[]', 'viscera[]', 'shell[]']:
                v_list = request.form.getlist(k)
                v = v_list[i] if i < len(v_list) else None
                if v is None or v.strip() == "":
                    return render_template('output.html', error="Please fill all fields.", results=None)
                vals.append(float(v))
            all_vals.append(vals)

        # Predict for each abalone entry
        for row in all_vals:
            X = np.array(row).reshape(1, -1)
            pred_rings = model.predict(X)[0]
            age = pred_rings + 1.5
            results.append({
                "rings": round(pred_rings, 2),
                "age": round(age, 2)
            })

        # Save the last predicted age
        if results:
            last_predicted_age = results[-1]["age"]

        return render_template('output.html', results=results, error=None)

    except Exception as e:
        return render_template('output.html', error=str(e), results=None)


if __name__ == '__main__':
    app.run(debug=True, port=8000)
