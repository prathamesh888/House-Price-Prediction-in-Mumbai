import pandas as pd
import numpy as np
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
df = pd.read_csv('CSVV.csv')
pipe = pickle.load(open("Modell.pkl", "rb"))

@app.route('/')
def index():

    locations =sorted(df['Location'].unique())
    return render_template('index1.html', locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    area = request.form.get('Area')
    location = request.form.get('location')
    bhk = request.form.get('BHK')
    new = request.form.get('New')
    gym = request.form.get('gym')
    lift = request.form.get('lift')
    car = request.form.get('Car')
    security = request.form.get('security')
    club = request.form.get('Club')
    pool = request.form.get('Pool')
    print(area, location, bhk, new)
    input = pd.DataFrame([[area, location, bhk, new, gym, lift, car, security, club, pool]], columns=['Area', 'Location', 'BHK', 'New/Resale', 'Gymnasium', 'Lift Available' , 'Car Parking', '24x7 Security', 'Clubhouse', 'Swimming Pool'])
    prediction = pipe.predict(input)[0]

    return str(np.round(prediction,2))


if __name__ == "__main__":
    app.run(debug=True, port=5001)
