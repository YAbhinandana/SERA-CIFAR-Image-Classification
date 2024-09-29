from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
model = load_model('cifar10_model.h5')
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']  
# Configure upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Route to the home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/report.html')
def rpt():
    return render_template('report.html')
@app.route('/about.html')
def about():
    return render_template('about.html')

@app.route('/Sdet.html')
def Sdet():
    return render_template('Sdet.html')

# Route to handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Preprocess the image for prediction
        image = load_img(filepath, target_size=(32, 32))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = image.astype('float32') / 255.0

        # Make a prediction
        predictions = model.predict(image)
        predicted_class = np.argmax(predictions, axis=1)[0]
        predicted_label = classes[predicted_class]

        return render_template('result.html', label=predicted_label, filename=filename)

    return redirect(url_for('index'))

# Route to display the uploaded image and prediction result
@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
