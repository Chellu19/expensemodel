import firebase_admin 
from firebase_admin import credentials, firestore
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflowjs as tfjs
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

cred = credentials.Certificate("firebase-admin-key.json") 
firebase_admin.initialize_app(cred)
db = firestore.client()

def fetch_expenses():
    print("🔄 Fetching expenses from Firestore...")
    expenses_ref = db.collection("expenses")
    docs = expenses_ref.stream()

    data = []
    for doc in docs:
        d = doc.to_dict()
        d['id'] = doc.id
        data.append(d)

    print(f"✅ {len(data)} records fetched")
    return pd.DataFrame(data)

df = fetch_expenses()

print("🧹 Preprocessing...")

required_columns = ['amount', 'participants', 'deadline']
df = df.dropna(subset=required_columns)

df['participant_count'] = df['participants'].apply(lambda x: len(x) if isinstance(x, list) else 0)
df['deadline_month'] = pd.to_datetime(df['deadline'].astype(str), errors='coerce').dt.month.fillna(0)

# Prepare features
X = df[['participant_count', 'deadline_month']].values
y = df['amount'].values

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data
print("📊 Normalizing data...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("🧠 Training model...")

# Build model with compatible architecture
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
history = model.fit(
    X_train_scaled, y_train, 
    epochs=50, 
    batch_size=32, 
    validation_split=0.1, 
    verbose=1
)

# Evaluate the model
test_loss, test_mae = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"📈 Test MAE: {test_mae:.2f}")

# Export directory
EXPORT_DIR = "tfjs_model"
os.makedirs(EXPORT_DIR, exist_ok=True)

print("📦 Exporting model...")

try:
    # Try with newer tensorflowjs version parameters
    tfjs.converters.save_keras_model(
        model, 
        EXPORT_DIR,
        skip_op_check=False,
        strip_debug_ops=True
    )
    print("✅ Model exported with advanced options")
except TypeError as e:
    print(f"⚠️ Advanced options not supported: {e}")
    print("🔄 Trying basic export...")
    
    # Fallback to basic export
    try:
        tfjs.converters.save_keras_model(model, EXPORT_DIR)
        print("✅ Model exported with basic options")
    except Exception as e:
        print(f"❌ Model export failed: {e}")
        raise

# Save scaler parameters
scaler_params = {
    'mean': scaler.mean_.tolist(),
    'scale': scaler.scale_.tolist(),
    'feature_names': ['participant_count', 'deadline_month']
}

scaler_path = os.path.join(EXPORT_DIR, 'scaler_params.json')
with open(scaler_path, 'w') as f:
    json.dump(scaler_params, f, indent=2)

# Save model metadata
metadata = {
    'input_shape': [None, 2],
    'output_shape': [None, 1],
    'features': ['participant_count', 'deadline_month'],
    'target': 'amount',
    'training_samples': len(X_train),
    'test_mae': float(test_mae),
    'model_version': '1.0'
}

metadata_path = os.path.join(EXPORT_DIR, 'model_metadata.json')
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"✅ Model exported to: {EXPORT_DIR}/")
print(f"📊 Scaler parameters saved: {scaler_path}")
print(f"📋 Model metadata saved: {metadata_path}")
print(f"📦 Upload contents of {EXPORT_DIR}/ to your hosting platform")

# Print model summary
print("\n📋 Model Summary:")
model.summary()

print(f"\n📊 Training completed:")
print(f"   • Training samples: {len(X_train)}")
print(f"   • Test samples: {len(X_test)}")
print(f"   • Test MAE: {test_mae:.2f}")
print(f"   • Input features: participant_count, deadline_month")

# Check if files were created successfully
files_created = []
for filename in ['model.json', 'scaler_params.json', 'model_metadata.json']:
    filepath = os.path.join(EXPORT_DIR, filename)
    if os.path.exists(filepath):
        size = os.path.getsize(filepath)
        files_created.append(f"   • {filename} ({size} bytes)")
    else:
        print(f"⚠️ Warning: {filename} not found!")

if files_created:
    print("\n📁 Files created:")
    for file_info in files_created:
        print(file_info)