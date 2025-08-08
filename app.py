from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import os
import yaml
from PIL import Image
import io
import base64
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

# Create Flask app
app = Flask(__name__)

# If everything is ok, print the parameters
print("[INFO] Configuration parameters...")
print("* Batch size........................:", params['batchSize'])
print("* Image height......................:", params['imgHeight'])
print("* Image width.......................:", params['imgWidth'])
print("* Number of epochs..................:", params['epochs'])
print("* Train dataset path................:", params['trainDatasetPath'])
print("* Train model path..................:", params['modelPath'])

# Load or train the model
if not os.path.exists(params['modelPath']):
    classification = Classification(params)
    model = classification.execute()
else:
    model = tf.keras.models.load_model(params['modelPath'])

# Load class names correctly
train_dataset = tf.keras.utils.image_dataset_from_directory(
    params['trainDatasetPath'],
    image_size=(params['imgHeight'], params['imgWidth']),
    batch_size=params['batchSize']
)
class_names = train_dataset.class_names
print(f"Loaded class names: {class_names}")


def preprocess_image(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize((params['imgWidth'], params['imgHeight']))
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = tf.expand_dims(image_array, 0)
    return image_array


@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    image_data = None
    if request.method == 'POST':
        if 'file' not in request.files or request.files['file'].filename == '':
            return render_template('index.html', prediction=None, image_data=None)

        file = request.files['file']
        image = Image.open(io.BytesIO(file.read()))

        # Preprocess the image and predict
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image)

        # Convert logits to probabilities
        score = tf.nn.softmax(predictions[0])

        # Get predicted class and confidence
        predicted_class = class_names[np.argmax(score)]
        confidence = np.max(score)

        prediction = {
            'class': predicted_class,
            'confidence': f"{confidence:.2%}"
        }

        # Convert image to base64 for display
        img_io = io.BytesIO()
        image.save(img_io, 'PNG')
        img_io.seek(0)
        image_data = base64.b64encode(img_io.getvalue()).decode()

    return render_template('index.html', prediction=prediction, image_data=image_data)


if __name__ == '__main__':
    app.run(debug=True, port=params['port'], host='0.0.0.0')