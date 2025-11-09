from flask import Flask, render_template, request #type: ignore
import tensorflow as tf #type: ignore
from tensorflow.keras.models import load_model #type: ignore
import numpy as np #type: ignore
import cv2 #type: ignore
import sqlite3
import os

# Configure TensorFlow for low memory usage
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

app = Flask(__name__)

# Function to create model architecture manually
def create_model():
    """Create the emotion detection model architecture"""
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(7, activation='softmax')
    ])
    return model

# Load model with compatibility handling
try:
    # Try loading with TF 2.x compatibility
    with tf.keras.utils.custom_object_scope({'InputLayer': tf.keras.layers.InputLayer}):
        model = tf.keras.models.load_model('face_emotionModel.h5', compile=False)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Attempting to load weights only...")
    try:
        # Create model architecture and load weights
        model = create_model()
        model.load_weights('face_emotionModel.h5')
        print("Model weights loaded successfully!")
    except Exception as e2:
        print(f"Error loading weights: {e2}")
        print("Creating new model (predictions may not be accurate without trained weights)")
        model = create_model()

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Emotion labels in FER2013 order
emotion_labels = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

# Create database if missing
conn = sqlite3.connect('database.db')
conn.execute('CREATE TABLE IF NOT EXISTS uploads (id INTEGER PRIMARY KEY AUTOINCREMENT, filename TEXT, emotion TEXT)')
conn.close()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    if file:
        filepath = os.path.join('static', file.filename)
        os.makedirs('static', exist_ok=True)
        file.save(filepath)

        # Load image
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (48,48))
        img = img.reshape(1,48,48,1) / 255.0

        # Predict emotion with reduced memory
        prediction = model.predict(img, verbose=0)
        emotion_index = np.argmax(prediction)
        emotion = emotion_labels[emotion_index]

        # Log to database
        conn = sqlite3.connect('database.db')
        conn.execute("INSERT INTO uploads (filename, emotion) VALUES (?, ?)", (file.filename, emotion))
        conn.commit()
        conn.close()

        return f"Predicted Emotion: {emotion}"
    return "No file uploaded"

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
