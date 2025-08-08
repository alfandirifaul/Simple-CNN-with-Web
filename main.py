print("[INFO] Import necessary libraries...")
import tensorflow as tf
import numpy as np
import io
import base64
import time
import os
import yaml
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
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

# Function to generate training plot
def generate_training_plot():
    # Check if training history exists
    history_path = 'training_history.npy'
    if os.path.exists(history_path):
        history = np.load(history_path, allow_pickle=True).item()
    else:
        # Generate dummy data for demonstration
        epochs = list(range(1, params['epochs'] + 1))
        history = {
            'accuracy': [0.3 + 0.7 * (1 - np.exp(-0.5 * i)) + np.random.normal(0, 0.02) for i in epochs],
            'val_accuracy': [0.25 + 0.65 * (1 - np.exp(-0.4 * i)) + np.random.normal(0, 0.03) for i in epochs],
            'loss': [2.0 * np.exp(-0.3 * i) + np.random.normal(0, 0.05) for i in epochs],
            'val_loss': [2.2 * np.exp(-0.25 * i) + np.random.normal(0, 0.07) for i in epochs]
        }

    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor('#0f172a')

    # Plot accuracy
    epochs = range(1, len(history['accuracy']) + 1)
    ax1.plot(epochs, history['accuracy'], 'o-', color='#06d6a0', label='Training Accuracy', linewidth=2)
    ax1.plot(epochs, history['val_accuracy'], 'o-', color='#f72585', label='Validation Accuracy', linewidth=2)
    ax1.set_title('Model Accuracy', color='white', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', color='#94a3b8')
    ax1.set_ylabel('Accuracy', color='#94a3b8')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor('#1e293b')

    # Plot loss
    ax2.plot(epochs, history['loss'], 'o-', color='#06d6a0', label='Training Loss', linewidth=2)
    ax2.plot(epochs, history['val_loss'], 'o-', color='#f72585', label='Validation Loss', linewidth=2)
    ax2.set_title('Model Loss', color='white', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', color='#94a3b8')
    ax2.set_ylabel('Loss', color='#94a3b8')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_facecolor('#1e293b')

    plt.tight_layout()

    # Convert plot to base64
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', bbox_inches='tight', facecolor='#0f172a', dpi=100)
    img_buffer.seek(0)
    plot_data = base64.b64encode(img_buffer.getvalue()).decode()
    plt.close()

    return plot_data

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

    # Generate training plot
    trainingPlot = generate_training_plot()

    # Check methods is POST?
    if request.method == 'POST':
        # Get the uploaded file from request
        if 'file' not in request.files or request.files['file'].filename == '':
            return render_template(
                'index.html',
                prediction=prediction,
                imageData=imageData,
                trainingPlot=trainingPlot
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
            'class'             : predictedClass,
            'confidence'        : f"{confidence:.2%}",
            'confidence_raw'    : f"{confidence * 100:.1f}",  # Raw percentage for progress bar
            'processing_time'   : f"{processTime:.3f}",
            'metadata'          : {
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
        imageData=imageData,
        trainingPlot=trainingPlot
    )

# Run the Flask app
if __name__ == '__main__':
    print("[INFO] Starting Flask app...")
    app.run(
        host='0.0.0.0',
        port=params['port'],
        debug=True
    )