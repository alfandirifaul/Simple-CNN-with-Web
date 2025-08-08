print("[INFO] Import necessary libraries...")
import tensorflow as tf
import numpy as np
import io
import base64
import time
import os
import yaml
from PIL import Image
from flask import Flask, render_template, request
from classification.ClassClassification import Classification

# First of all, read parameters from configuration file
assert os.path.exists('configuration.yaml'), "Error: configuration.yaml file not found."
with open('configuration.yaml') as file:
    params = yaml.safe_load(file)

# Check the parameters
assert params['trainDatasetPath'], "Error: 'trainDatasetPath' not found in configuration.yaml."
assert params['modelPath'], "Error: 'modelPath' not found in configuration.yaml."
assert params['batchSize'], "Error: 'batchSize' not found in configuration.yaml."
assert params['epochs'], "Error: 'epochs' not found in configuration.yaml."
assert params['imgWidth'] == params['imgHeight'], "Error: 'imgWidth' and 'imgHeight' must be equal in configuration.yaml."
assert params['port'], "Error: 'port' not found in configuration.yaml."

# If everything is ok, print the parameters
print("[INFO] Configuration parameters...")
print("* Batch size........................:", params['batchSize'])
print("* Image height......................:", params['imgHeight'])
print("* Image width.......................:", params['imgWidth'])
print("* Number of epochs..................:", params['epochs'])
print("* Train dataset path................:", params['trainDatasetPath'])
print("* Train model path..................:", params['modelPath'])

# Create Flask app
app = Flask(__name__)

# Initialize the Classification class with parameters
classification = Classification(params)

# Get the model
model = classification.isModelTrained()

# Load class names
_, _, classNames, _ = classification.loadDataset()

# Function to preprocess the image after uploading
def imagePreprocess(image):
    # Check if the image is in RGB format
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Resize the image to the required dimensions
    image = image.resize(
        (params['imgWidth'], params['imgHeight']),
    )

    # Convert image to array
    imgArr = tf.keras.preprocessing.image.img_to_array(image)
    imgArr = tf.expand_dims(imgArr, 0)

    # Return the preprocessed image
    return imgArr

# Function to convert the size images to byte
def formatBytes(size):
    # Convert bytes to a human-readable format
    power = 1024
    n = 0
    label = {
        0: "B",
        1: "KB",
        2: "MB",
        3: "GB",
        4: "TB",
    }

    # Loop to find the appropriate label
    while size >= power and n < len(label) - 1:
        size /= power
        n += 1

    # Return the formatted size
    return f"{size:.2f} {label[n]}"

# Route for the home page
@app.route('/', methods=['GET', 'POST'])
def home():

    # Define variables for the template
    prediction = None
    imageData = None

    # Check methods is POST?
    if request.method == 'POST':
        # Get the uploaded file from request
        if 'file' not in request.files or request.files['file'].filename == '':
            return render_template(
                'index.html',
                prediction=prediction,
                imageData=imageData,
            )

        # Start the timer and note it.
        startTime = time.time()

        # Get the file from the request
        file = request.files['file']

        # Read the file content for processing
        fileBytes = file.read()

        # Get the metadata of the image
        fileSizeInBytes = len(fileBytes)
        formattedSize   = formatBytes(fileSizeInBytes)

        # Get the image size and dimensions
        image = Image.open(
            io.BytesIO(fileBytes)
        )
        width, height   = image.size
        dimensions      = f"{width} x {height} pixels"

        # Predict the class of the image
        processedImage  = imagePreprocess(image)
        predictions     = model.predict(processedImage)
        score           = tf.nn.softmax(predictions[0])
        predictedClass  = classNames[np.argmax(score)]
        confidence      = np.max(score)

        # End the timer and note it.
        endTime = time.time()

        # Calculate the processing time
        processTime = endTime - startTime

        # Prepare the prediction dictionary with the new data
        prediction = {
            'class'         : predictedClass,
            'confidence'    : f"{confidence:.2%}",
            'processingTime': f"{processTime:.3f}",
            'metadata'      : {
                'dimensions'    : dimensions,
                'size'          : formattedSize,
            }
        }

        # Convert the image to base64 for rendering in template
        imgIo = io.BytesIO()
        image.save(
            imgIo,
            format="PNG"
        )
        imgIo.seek(0)
        imageData = base64.b64encode(imgIo.getvalue()).decode()

    # Render the template with the prediction and image data
    return render_template(
        'index.html',
        prediction=prediction,
        imageData=imageData
    )

# Run the Flask app
if __name__ == '__main__':
    print("[INFO] Starting Flask app...")
    app.run(
        host='0.0.0.0',
        port=params['port'],
        debug=True
    )






