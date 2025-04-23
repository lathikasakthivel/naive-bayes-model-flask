from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import os
from io import StringIO
from model.naive_bayes_model import train_model, classify_instance

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure uploads directory exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        try:
            train_file = request.files['file']
            instance_data = request.form['instance']

            if train_file:
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], train_file.filename)
                train_file.save(filepath)

                if filepath.endswith('.csv'):
                    df = pd.read_csv(filepath)
                else:
                    df = pd.read_excel(filepath)

                model = train_model(df)

                test_data = [instance_data.strip().split(',')]
                prediction = classify_instance(model, test_data)

                return render_template('result.html', prediction=prediction)
        except Exception as e:
            return f"Error: {str(e)}"
    return render_template('upload.html')

@app.route('/manual', methods=['GET', 'POST'])
def manual_entry():
    if request.method == 'POST':
        try:
            train_data = request.form['data']
            instance_data = request.form['instance']

            rows = [row.split(',') for row in train_data.strip().split('\n')]
            df = pd.DataFrame(rows[1:], columns=rows[0])
            df = df.apply(pd.to_numeric, errors='ignore')

            model = train_model(df)

            # Process instance input (DON'T convert to float — just split)
            test_data = [instance_data.strip().split(',')]
            prediction = classify_instance(model, test_data)


            return render_template('result.html', prediction=prediction)
        except Exception as e:
            return f"Error: {str(e)}"
    return render_template('manual_entry.html')

# Error log for 500 errors (helpful on Render)
@app.errorhandler(500)
def internal_error(error):
    import traceback
    print(traceback.format_exc())
    return "Internal Server Error Occurred", 500

if __name__ == '__main__':
    app.run(debug=True)
