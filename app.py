from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('ndvi_model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        ndvi = float(request.form['ndvi'])
        prediction = model.predict(np.array([[ndvi]]))[0]
        class_map = {
            0: "Water or Urban Area",
            1: "Barren Land",
            2: "Sparse Vegetation / Grassland",
            3: "Unhealthy Crops",
            4: "Thriving Crops",
            5: "Rainforest or Dense Vegetation"
        }
        label = class_map.get(prediction, "Unknown")
        return render_template('index.html', result=f"NDVI class: {label}")
    except Exception as e:
        return render_template('index.html', result=f"Error: {e}")

@app.route('/map')
def map():
    return render_template('map.html')
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
