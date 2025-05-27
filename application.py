from flask import Flask, render_template, request, session
import numpy as np
import pandas as pd

from src.pipeline.inference_pipeline import CustomData, InferencePipeline

application= Flask(__name__)

app= application
app.secret_key = "secret_key"

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if 'history' not in session:
        session['history'] = []

    if request.method == 'POST':
        review = request.form.get('review')
        if review.strip():
            data = CustomData(review=review)
            pred_df = data.get_data_as_dataframe()

            inference_pipeline = InferencePipeline()
            results = inference_pipeline.predict(features=pred_df)
            prediction = results[0]

            session['history'].append({'review': review, 'prediction': prediction})
            session.modified = True

    return render_template('home.html', history=session.get('history', []))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
