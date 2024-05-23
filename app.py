import os
from flask import Flask, request, render_template, redirect, url_for
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import cv2
from werkzeug.utils import secure_filename

print("loading the model")
app = Flask(__name__)

# Load the model
model = load_model('model/CNN_model.h5')

# Set the upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def preprocess_image(img_path,target_size=(50, 50)):
    print("Preprocessing image:", img_path)  # Debugging statement
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB format
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)  # Resize to your target size
    img_array = np.expand_dims(img, axis=0)  # Add batch dimension
    return img_array

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            print("No file part in the request.")  # Debugging statement
            return redirect(request.url)
        file = request.files['file']
        # If user does not select file, browser also submits an empty part without filename
        if file.filename == '':
            print("No selected file.")  # Debugging statement
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Preprocess the image and make prediction
            img_array = preprocess_image(file_path)
            prediction = model.predict(img_array)[0][0]
            result = "Cancer Detected" if prediction > 0.5 else "No Cancer Detected"
            
            print("Prediction result:", result)  # Debugging statement

            return render_template('index1.html', filename=filename, result=result)
    return render_template('index1.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == '__main__':
    app.run(debug=True)

