
import cv2
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt

import skimage.io as io
from skimage.transform import resize
from skimage.io import imread

from tensorflow import keras
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD, Adam, Adamax
from tensorflow.keras.preprocessing import image

# Suppress TensorFlow warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def get_model_info():
    """Return information about available models and datasets"""
    return {
        'datasets': ['HuSHeM', 'SMIDS'],
        'models': ['Xception', 'MobileNet', 'GoogleNet'],
        'HuSHeM_classes': ['01_Normal', '02_Tapered', '03_Pyriform', '04_Amorphous'],
        'SMIDS_classes': ['Acrosome Abnormality', 'Boya', 'Sperm']
    }


def test_model(dataset, model, directory):

    if(len(directory) > 0):
        img_size = 0
        classes = []
        opt = ''
        model_path = ''
        
        # Updated image sizes and class names to match the trained models
        if(dataset == 'HuSHeM'):
            img_size = 170  # Updated from 131 to 170 to match training
            classes = ['01_Normal', '02_Tapered', '03_Pyriform', '04_Amorphous']  # Added prefixes
            opt = 'adam'  # Use adam optimizer for better performance
            # Path to the HuSHeM trained model
            model_path = '../HuSHeM-20250630T085035Z-1-001/HuSHeM/'
        else:  # SMIDS dataset
            img_size = 170
            classes = ['Acrosome Abnormality', 'Boya', 'Sperm']  # Fixed class names
            opt = 'adamax'  # Use adamax for SMIDS
            # Path to the SMIDS dataset (using transfer learning model)
            model_path = '../HuSHeM-20250630T085035Z-1-001/HuSHeM/'  # Transfer learning uses HuSHeM base model

        # Load the trained model files with correct paths and names
        try:
            # Load JSON model architecture
            json_file_path = os.path.join(model_path, 'mobil.json')
            with open(json_file_path, 'r') as json_file:
                loaded_model_json = json_file.read()

            # Create model from JSON
            loaded_model = model_from_json(loaded_model_json)

            # Load weights with correct .weights.h5 extension
            weights_file_path = os.path.join(model_path, 'Mobil_Hushem.weights.h5')
            loaded_model.load_weights(weights_file_path)

            # Compile model with appropriate optimizer
            if opt == 'adam':
                from tensorflow.keras.optimizers import Adam
                optimizer = Adam(learning_rate=0.0001)
            elif opt == 'adamax':
                from tensorflow.keras.optimizers import Adamax
                optimizer = Adamax(learning_rate=0.0001)
            else:
                from tensorflow.keras.optimizers import SGD
                optimizer = SGD(learning_rate=0.001)
                
            loaded_model.compile(loss="categorical_crossentropy",
                                optimizer=optimizer, metrics=['accuracy'])

            # Read and preprocess the image
            img = io.imread(directory)
            img = resize(img, (img_size, img_size, 3), anti_aliasing=True)
            image = np.asarray(img)
            image = np.expand_dims(image, axis=0)

            # Make prediction
            pred = loaded_model.predict(image, verbose=0)

            # Get the class with highest probability
            y_pred = np.argmax(pred)

            # Format prediction results
            pred_flat = pred.flatten()
            
            # Create formatted text with percentages
            text = ""
            total = np.sum(pred_flat)
            
            for n in range(len(classes)):
                percentage = (pred_flat[n] / total) * 100
                text += f"{classes[n]}: {percentage:.2f}%\n"

            return text, classes[y_pred]
            
        except FileNotFoundError as e:
            error_msg = f"Model files not found. Please ensure the trained models are in the correct location.\nError: {str(e)}"
            return error_msg, "Error"
        except Exception as e:
            error_msg = f"Error loading model or processing image: {str(e)}"
            return error_msg, "Error"
    else:
        return "Please specify the image path", "No Image Selected"
