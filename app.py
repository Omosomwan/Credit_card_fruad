import joblib
from flask import Flask, request, render_template
import matplotlib.pyplot as plt
from matplotlib.image import imread

app = Flask(__name__)
model = joblib.load('prediction.joblib')

@app.route("/")

def home():
    return render_template('form.html')
    

@app.route("/predict", methods = ["GET", "POST"])

def prediction():
    result = ''

    if request.method == "POST":
        features = [
            float(request.form.get("amount")),
            float(request.form.get("transaction_hour")),
            float(request.form.get("foreign_transaction")),
            float(request.form.get("location_mismatch")),
            float(request.form.get("device_trust_score")),
            float(request.form.get("velocity_last_24h")),
            float(request.form.get("cardholder_age")),
        ]
        y_pred = model.predict([features])[0]
    
    if y_pred == 1:
        result = "It is a fraud"
    else:
        result = "It is not a fraud"

    return render_template("form.html", result = result)


if __name__ == '__main__':
    app.run(debug = True)
