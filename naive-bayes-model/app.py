from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import os
from io import StringIO
from model.naive_bayes_model import train_model, classify_instance

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        train_file = request.files['file']
        instance_data = request.form['instance']
        if train_file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], train_file.filename)
            train_file.save(filepath)
            df = pd.read_csv(filepath) if train_file.filename.endswith('.csv') else pd.read_excel(filepath)

            # Train model and save
            model = train_model(df)

            # Process instance input
            test_data = np.array([list(map(float, instance_data.strip().split(',')))])
            prediction = classify_instance(model, test_data)
            return render_template('result.html', prediction=prediction)
    return render_template('upload.html')

@app.route('/manual', methods=['GET', 'POST'])
def manual_entry():
    if request.method == 'POST':
        train_data = request.form['data']
        instance_data = request.form['instance']

        rows = [row.split(',') for row in train_data.strip().split('\n')]
        df = pd.DataFrame(rows[1:], columns=rows[0])
        df = df.apply(pd.to_numeric, errors='ignore')

        model = train_model(df)
        test_data = np.array([list(map(float, instance_data.strip().split(',')))])
        prediction = classify_instance(model, test_data)

        return render_template('result.html', prediction=prediction)
    return render_template('manual_entry.html')

if __name__ == '__main__':
    app.run(debug=True)
