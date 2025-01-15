from flask import Flask, request, render_template, send_from_directory
from werkzeug.utils import secure_filename
import numpy as np
import cv2
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
import logging
import os

# Initialize Flask app
app = Flask(__name__)

# Configure upload folder
app.config['UPLOAD_FOLDER'] = 'static/uploads'
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the models
try:
    with open('svm_model_optimized.pkl', 'rb') as f:
        svm_model = pickle.load(f)
    resnet_model = load_model('resnet50_model.h5')
    logger.info("Models loaded successfully")
except Exception as e:
    logger.error(f"Error loading models: {str(e)}")
    raise

Categories = ['VI-shingles', 'VI-chickenpox', 'BA-cellulitis', 'FU-athlete-foot', 
              'BA-impetigo', 'FU-nail-fungus', 'FU-ringworm', 'PA-cutaneous-larva-migrans','Healthy']
img_size = (224, 224)

def preprocess_image(image):
    try:
        if image is None:
            raise ValueError("Input image is None")

        # Ensure image is in uint8 format
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        # Resize
        image = cv2.resize(image, img_size)
        
        # Contrast Enhancement (CLAHE)
        try:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            lab = cv2.merge([l, a, b])
            image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        except Exception as e:
            logger.warning(f"CLAHE enhancement failed, using original image. Error: {str(e)}")
        
        # Convert to float32 and preprocess for ResNet50
        image = image.astype(np.float32)
        image = preprocess_input(image)
        
        return image

    except Exception as e:
        logger.error(f"Error in image preprocessing: {str(e)}")
        raise

def validate_image(file):
    if not file:
        raise ValueError("No file uploaded")
    
    if file.filename == '':
        raise ValueError("No file selected")
    
    allowed_extensions = {'png', 'jpg', 'jpeg', 'bmp', 'webp'}
    if not '.' in file.filename or \
       file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
        raise ValueError("Invalid file type. Please upload a valid image file.")
    
    return True

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return render_template('result.html', error='No file uploaded')
        
        file = request.files['file']
        validate_image(file)
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        img = cv2.imread(filepath)
        if img is None:
            return render_template('result.html', error='Could not read the image file')
        
        # Preprocess image
        processed_img = preprocess_image(img)
        processed_img = np.expand_dims(processed_img, axis=0)
        
        # Extract features using ResNet
        features = resnet_model.predict(processed_img, verbose=0)
        features_flat = features.reshape(1, -1)
        
        # Make prediction using SVM
        prediction = svm_model.predict(features_flat)[0]
        # Get probability scores if your SVM model was trained with probability=True
        try:
            probabilities = svm_model.predict_proba(features_flat)[0]
            confidence = round(float(np.max(probabilities)) * 100, 2)
        except:
            # If probabilities are not available, set confidence to None
            confidence = None
        
        predicted_label = Categories[prediction]
        if '-' in predicted_label:
            disease_name = predicted_label.split('-')[1].replace('-', ' ').title()
            category = predicted_label.split('-')[0]
        else:
            disease_name = predicted_label
            category = ""
        
        return render_template('result.html',
                             prediction=disease_name,
                             category=category,
                             confidence=confidence,
                             image_filename=filename)

    except ValueError as ve:
        logger.warning(f"Validation error: {str(ve)}")
        return render_template('result.html', error=str(ve))
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return render_template('result.html', 
                             error='An error occurred during prediction. Please try again.')

@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
