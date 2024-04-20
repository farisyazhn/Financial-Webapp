import numpy as np
from flask import Flask, render_template, request
import pickle
import csv 

app = Flask(__name__)
app.secret_key = 'dx8bx95x15xd4r1xd3axbekxaaxd6xd5@Ax1fLSQ8$CLCPxe4(x88xccxx9a'

# Load the pickled model from a separate module or file
with open('model.pkl', 'rb') as f:
    pickled_model = pickle.load(f)
    
# Use a dictionary to map the categorical variables to numerical variables
categorical_mapping = {
    'Male': 1,
    'Female': 0,
    'Yes': 1,
    'No': 0,
    'Other': 5,
    'Working': 4,
    'Student': 3,
    'State servant': 2,
    'Pensioner': 1,
    'Commercial associate': 0,
    'Secondary / secondary special': 4,
    'Lower secondary': 3,
    'Incomplete higher': 2,
    'Higher education': 1,
    'Academic degree': 0,
    'Widow': 4,
    'Single / not married': 3,
    'Separated': 2,
    'Married': 1,
    'Civil marriage': 0,
    'With parents': 5,
    'Rented apartment': 4,
    'Office apartment': 3,
    'Municipal apartment': 2,
    'House / apartment': 1,
    'Co-op apartment': 0,
}

def preprocess_input(feature, value):
    # Use input validation to ensure that the user inputs valid values for the input features
    if feature == 'Total_Children' or feature == 'Total_Family_Members':
        return int(value)
    elif feature == 'Total_Income' or feature == 'Applicant_Age' or feature == 'Years_of_Working' or feature == 'Total_Bad_Debt' or feature == 'Total_Good_Debt':
        return int(value)
    else:
        return categorical_mapping.get(value, 0)

@app.route('/')
def index():
    return render_template('index.html')

def save_to_csv(input_features, prediction_result):
    with open('credit_card_outcome.csv', mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([*input_features.values(), prediction_result])

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            input_features = {}
            for key, value in request.form.items():
                input_features[key] = preprocess_input(key, value)
            
            # Create a NumPy array from the preprocessed input features
            arr = np.array([list(input_features.values())])

            # Predict the output using the pre-trained model
            pred = pickled_model.predict(arr)
            
            # Handle the model prediction more robustly
            if pred == 0:
                result = "You are not eligible for a credit card."
            elif pred == 1:
                result = "You are eligible for a credit card."
            # Save to csv 
            save_to_csv(input_features, result) 

            # Render the results template and pass the prediction result to it
            return render_template('results.html', prediction_text=result)

        except Exception as e:
            # Add error handling for invalid input values
            error_message = str(e)
            return render_template('error.html', error_text=error_message)

if __name__ == "__main__":
    app.run(debug=True)