from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load('model/scam_model.pkl')  # Ensure model path is correct

@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    if request.method == 'POST':
        try:
            # Collect form values
            features = [
                int(request.form['username_length']),
                int(request.form['username_has_number']),
                int(request.form['full_name_has_number']),
                int(request.form['full_name_length']),
                int(request.form['is_private']),
                int(request.form['is_joined_recently']),
                int(request.form['has_channel']),
                int(request.form['is_business_account']),
                int(request.form['has_guides']),
                int(request.form['has_external_url']),
                int(request.form['edge_followed_by']),  # followers
                int(request.form['edge_follow'])         # followings
            ]

            # Convert to DataFrame to match training format
            columns = [
                'username_length', 'username_has_number', 'full_name_has_number', 'full_name_length',
                'is_private', 'is_joined_recently', 'has_channel', 'is_business_account',
                'has_guides', 'has_external_url', 'edge_followed_by', 'edge_follow'
            ]

            features_df = pd.DataFrame([features], columns=columns)

            # Make prediction
            prediction = model.predict(features_df)[0]

            # Display result
            if prediction == 1:
                result = "❌ Fake Account"
            else:
                result = "✅ Real Account"

        except Exception as e:
            result = f"Error: {str(e)}"

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
