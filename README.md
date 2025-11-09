# Facial Emotion Detection System

A web-based facial emotion detection application using deep learning to classify emotions from uploaded images.

## Features

- 🎭 Detects 7 emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
- 📷 Easy image upload with drag-and-drop support
- 💅 Beautiful modern UI with responsive design
- 🚀 Fast emotion prediction using CNN model

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

## Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd FACE_DETECTION
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On Mac/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the trained model**
   
   The model file (`face_emotionModel.h5`) is too large for Git. You need to:
   
   - **Option 1:** Train your own model using `model_training.py` (requires the FER2013 dataset)
   - **Option 2:** Download from [Google Drive/Dropbox link] (add your link here)
   - **Option 3:** Use Git LFS (see below)
   
   Place the `face_emotionModel.h5` file in the root directory.

## Usage

1. **Start the application**
   ```bash
   python app.py
   ```

2. **Open your browser**
   
   Navigate to `http://127.0.0.1:5000`

3. **Upload an image**
   
   Click or drag-and-drop an image containing a face, then click "Detect My Emotion"

## Project Structure

```
FACE_DETECTION/
│
├── app.py                  # Main Flask application
├── model_training.py       # Model training script
├── requirements.txt        # Python dependencies
├── face_emotionModel.h5   # Trained model (not in repo)
├── database.db            # SQLite database for logs
│
├── templates/
│   └── index.html         # Frontend UI
│
├── static/                # Uploaded images
│
└── data/
    └── fer2013.csv        # Training dataset (not in repo)
```

## Deployment Options

### Option 1: Deploy to Render (Free)

1. **Create a `render.yaml` file** (already included in repo)
2. Go to [Render.com](https://render.com)
3. Sign up/login with your GitHub account
4. Click "New +" → "Web Service"
5. Connect your GitHub repository
6. Render will auto-detect the settings
7. Click "Create Web Service"

**Note:** Upload your model file to a cloud storage service (Google Drive, Dropbox) and modify `app.py` to download it on startup.

### Option 2: Deploy to Railway (Free)

1. Go to [Railway.app](https://railway.app)
2. Sign up with GitHub
3. Click "New Project" → "Deploy from GitHub repo"
4. Select your repository
5. Railway will automatically deploy

### Option 3: Deploy to Heroku

1. Install Heroku CLI
2. Login to Heroku:
   ```bash
   heroku login
   ```
3. Create a new app:
   ```bash
   heroku create your-app-name
   ```
4. Deploy:
   ```bash
   git push heroku main
   ```

### Option 4: Deploy to PythonAnywhere

1. Sign up at [PythonAnywhere.com](https://www.pythonanywhere.com)
2. Upload your code via their file browser
3. Create a new web app
4. Configure WSGI file to point to your Flask app
5. Reload the web app

## Handling Large Files (Model)

### Method 1: Git LFS (Recommended)

```bash
# Install Git LFS
git lfs install

# Track large files
git lfs track "*.h5"
git add .gitattributes
git add face_emotionModel.h5
git commit -m "Add model with Git LFS"
git push
```

### Method 2: Cloud Storage

Upload your model to:
- Google Drive
- Dropbox
- AWS S3
- GitHub Releases

Then modify `app.py` to download the model on first run.

### Method 3: Model Download Script

Create a `download_model.py` script:

```python
import gdown
import os

# Google Drive file ID
file_id = "YOUR_FILE_ID_HERE"
output = "face_emotionModel.h5"

if not os.path.exists(output):
    print("Downloading model...")
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output, quiet=False)
    print("Model downloaded successfully!")
```

Run before starting the app:
```bash
python download_model.py
python app.py
```

## Environment Variables (for Deployment)

Create a `.env` file:
```
FLASK_ENV=production
SECRET_KEY=your-secret-key-here
PORT=5000
```

## Technologies Used

- **Backend:** Flask (Python)
- **Deep Learning:** TensorFlow/Keras
- **Computer Vision:** OpenCV
- **Frontend:** HTML, CSS, JavaScript
- **Database:** SQLite

## Model Information

- **Architecture:** Convolutional Neural Network (CNN)
- **Input:** 48x48 grayscale images
- **Output:** 7 emotion classes
- **Dataset:** FER2013

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License.

## Acknowledgments

- FER2013 dataset for emotion recognition
- TensorFlow/Keras for deep learning framework
- Flask for web framework

## Contact

Your Name - somadetoluwani@gmail.com
Project Link: [https://github.com/Adesmith001/FACE_DETECTION.git](https://github.com/Adesmith001/FACE_DETECTION.git)

---

**Note:** Remember to replace placeholder links and information with your actual details!
