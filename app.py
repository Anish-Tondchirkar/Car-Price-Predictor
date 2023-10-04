
from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import pandas as pd  



# Create a Flask application
app = Flask(__name__)



model = pickle.load(open('Random_Forest_model.pkl','rb'))

# Load the StandardScaler object (replace 'scaler.pkl' with the actual path)
with open('scaler.pkl', 'rb') as scaler_file:
    scalar = pickle.load(scaler_file)


# Define the indices of features to be standardized (year, km, mileage, engine, power, seats)
standardize_indices = [0, 1, 3, 4, 5, 6]

# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for making predictions
@app.route('/predict', methods=['POST'])
def predict_price():

        
    try:
        if request.method == 'POST':
            # Get input data from the request
            year = int(request.form['year'])
            km_driven = int(request.form['km_driven'])
            owner_type = request.form['owner_type']
            mileage = float(request.form['mileage'])
            engine = int(request.form['engine'])
            power = float(request.form['power'])
            seats = int(request.form['seats'])
            transmission = request.form['transmission']
            fuel_type = request.form['fuel_type']



            # Fuel
            
            if fuel_type == 'Petrol':
                Fuel_Type_Petrol = 1
                Fuel_Type_Diesel = 0
                Fuel_Type_LPG = 0
                Fuel_Type_CNG = 0
                Fuel_Type_Electric = 0
            elif fuel_type == 'Diesel':
                Fuel_Type_Petrol = 0
                Fuel_Type_Diesel = 1
                Fuel_Type_LPG = 0
                Fuel_Type_CNG = 0
                Fuel_Type_Electric = 0
            elif fuel_type == 'LPG':
                Fuel_Type_Petrol = 0
                Fuel_Type_Diesel = 0
                Fuel_Type_LPG = 1
                Fuel_Type_CNG = 0
                Fuel_Type_Electric = 0
            elif fuel_type == 'Electric':
                Fuel_Type_Petrol = 0
                Fuel_Type_Diesel = 0
                Fuel_Type_LPG = 0
                Fuel_Type_CNG = 0
                Fuel_Type_Electric = 1
            else:
                Fuel_Type_Petrol = 0
                Fuel_Type_Diesel = 0
                Fuel_Type_LPG = 0
                Fuel_Type_CNG = 0
                Fuel_Type_Electric = 0

            # Transmission
            if(transmission == 'Manual'):
                Transmission_Manual = 1
            else:
                Transmission_Manual = 0

            # Owner type
            if(owner_type == 'First'):
                Owner_Type = 1
            elif(owner_type == 'Second'):
                Owner_Type = 2
            elif(owner_type == 'Third'):
                Owner_Type = 3
            else:
                Owner_Type = 4

            input_data = np.array([[year, km_driven, owner_type,mileage, engine,
                                     power, seats, Transmission_Manual,
                                     Fuel_Type_Diesel, Fuel_Type_Electric, 
                                     Fuel_Type_LPG,Fuel_Type_Petrol ]])
            input_data[:, standardize_indices] = scalar.transform(input_data[:, standardize_indices])

            # Making Predictions using the model

            predicted_price = model.predict(input_data)[0]

            # Output Printing for user
            output = round(predicted_price,2)

            if output<0:
                return render_template('index.html', prediction_text="Sorry, you cannot sell this car")
            else:
                return render_template('index.html', prediction_text="You can sell the car at {}".format(output))
        else:
            return render_template('index.html')

    except Exception as e:
        return render_template('index.html', prediction_text="An error occurred: {}".format(str(e)))

# if __name__ == '__main__':
#     app.run(debug=True)
# This above is when running on my pc

# Below for uploading
if __name__ == '__main__':
    app.run(host='0.0.0.0',port=6969)





          
