import os
import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import preprocess_input

app = Flask(_name_)

# Load your trained model
model = load_model('model.h5')  # Adjust the path if necessary

UPLOAD_FOLDER = 'uploads'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/result', methods=["GET", "POST"])
def res():
    if request.method == "POST":
        # Check if the post request has the file part
        if 'image' not in request.files:
            return render_template('index.html', prediction='No file uploaded')

        file = request.files['image']

        # If user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return render_template('index.html', prediction='No file selected')

        if file:
            # Save the file to the uploads folder
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            # Load and preprocess the image
            img = image.load_img(filepath, target_size=(299, 299))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            img_data = preprocess_input(x)

            # Make prediction
            prediction = model.predict(img_data)
            result = np.argmax(prediction, axis=1)[0]

            # Define your result labels based on your model output
            index = ['COVID', 'Lung_Capacity', 'Normal', 'Viral Pneumonia']
            result_label = index[result]

            return render_template('index.html', prediction=result_label)

if _name_ == "_main_":
    app.run(debug=False)