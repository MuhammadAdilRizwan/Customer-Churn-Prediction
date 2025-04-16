from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
app = Flask(__name__)

# Load the pre-trained Random Forest model
with open(r'C:\Users\11th Generation\Desktop\Infinity_Cauldron\Customer_Churn_Project\Churn-Prediction_Deploy\RF_MODEL_FINALE.pkl', 'rb') as model_file:
    model = pickle.load(model_file)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        age = float(request.form['age'])
        gender = request.form['gender']
        tenure = float(request.form['tenure'])
        usage_frequency = float(request.form['usage_frequency'])
        support_calls = int(request.form['support_calls'])
        payment_delay = float(request.form['payment_delay'])
        total_spend = float(request.form['total_spend'])
        last_interaction = float(request.form['last_interaction'])
        subscription_contract = int(request.form['subscription_contract'])

        # Encode gender to match the training data
        gender_encoded = 1 if gender == 'Male' else 0

        # Create a feature array
        input_data = {
            'Age': [age],
            'Gender': [gender],
            'Tenure': [tenure],
            'Usage Frequency': [usage_frequency],
            'Support Calls': [support_calls],
            'Payment Delay': [payment_delay],
            'Total Spend': [total_spend],
            'Last Interaction': [last_interaction],
            'Subscription_Contract': [subscription_contract]
        }


        prediction = model.predict(pd.DataFrame(input_data))

        # Render the result page with dynamic background
        result_image = 'willchurn.jpg' if prediction == 1 else 'wontchurn.jpg'
        result_text = "The cosmic winds have shifted, and the data reveals a dark truth — churn is predicted. This customer may soon fade into the void, leaving your service behind." if prediction == 1 else  "The stars align in favor of your service. The data indicates the customer is bound to stay, their loyalty continuing under the cosmic forces of retention."
        return render_template('result.html', result_text=result_text, result_image=result_image)

    except Exception as e:
        return render_template('index.html', error="Error processing the input: " + str(e))


if __name__ == '__main__':
    app.run(debug=True)
