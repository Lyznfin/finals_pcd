import pandas as pd
import joblib

from utils.flatten import flatten_columns
from extract import extract

knn = joblib.load('models_strawberry/knn_model.pkl')
label_encoder = joblib.load('models_strawberry/label_encoder.pkl')
scaler = joblib.load('models_strawberry/scaler.pkl')

def classify_strawberry(image_path):
    feature_columns = ['contrast', 'homogeneity', 'energy', 'correlation']

    image_features = extract(image_path)

    image_features_df = pd.DataFrame([image_features], columns=feature_columns)
    flattened_df = flatten_columns(image_features_df)

    new_features_scaled = scaler.transform(flattened_df)
    predicted_class_encoded = knn.predict(new_features_scaled)
    predicted_class = label_encoder.inverse_transform(predicted_class_encoded)

    print(f'Prediksi mata: {predicted_class}')

# classify_strawberry('C:/Users/manzi/VSCoding/0_pcd/dataset/normal/NL_004.png')