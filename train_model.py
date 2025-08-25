import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the dataset
df = pd.read_csv("final-v1.csv")

# Feature columns used for training
feature_columns = [
    'username_length', 'username_has_number', 'full_name_has_number', 'full_name_length',
    'is_private', 'is_joined_recently', 'has_channel', 'is_business_account',
    'has_guides', 'has_external_url', 'edge_followed_by', 'edge_follow'
]

# Separate fake and real accounts
df_fake = df[df['is_fake'] == 1]
df_real = df[df['is_fake'] == 0]

# Upsample real accounts to balance the dataset
df_real_upsampled = resample(df_real,
                             replace=True,
                             n_samples=len(df_fake),
                             random_state=42)

# Combine the balanced dataset
df_balanced = pd.concat([df_fake, df_real_upsampled])

# Shuffle the rows (recommended)
df_balanced = df_balanced.sample(frac=1, random_state=42)

# Split features (X) and target (y)
X = df_balanced[feature_columns]
y = df_balanced['is_fake']

# Train the Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Save the trained model
joblib.dump(model, 'model/scam_model.pkl')

print("✅ Model trained and saved successfully.")
