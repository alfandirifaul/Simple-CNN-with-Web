from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import os
import yaml
from PIL import Image
import io
import base64
import time
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

# Initialize the Classification class with parameters
classification = Classification(params)

# Load or train the model
if not os.path.exists(params['modelPath']):
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

def format_bytes(size):
    """Mengubah byte menjadi format KB, MB, GB, atau TB yang mudah dibaca."""
    power = 1024
    n = 0
    power_labels = {0: 'B', 1: 'KB', 2: 'MB', 3: 'GB', 4: 'TB'}
    while size >= power and n < len(power_labels) - 1:
        size /= power
        n += 1
    return f"{size:.2f} {power_labels[n]}"


@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    image_data = None
    if request.method == 'POST':
        # Pastikan file ada dalam request
        if 'file' not in request.files or request.files['file'].filename == '':
            return render_template('index.html', prediction=None, image_data=None)

        # === MULAI PENGUKURAN ===
        start_time = time.time()  # Catat waktu mulai

        file = request.files['file']

        # Baca konten file ke memori untuk diolah
        file_bytes = file.read()

        # 3. Dapatkan metadata gambar
        # Ukuran file dari byte yang dibaca
        file_size_bytes = len(file_bytes)
        formatted_size = format_bytes(file_size_bytes)

        # Buka gambar menggunakan Pillow dan dapatkan dimensi
        image = Image.open(io.BytesIO(file_bytes))
        width, height = image.size
        dimensions = f"{width} x {height} pixels"

        # --- Lakukan Prediksi seperti biasa ---
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image)
        score = tf.nn.softmax(predictions[0])
        predicted_class = class_names[np.argmax(score)]
        confidence = np.max(score)

        # === SELESAI PENGUKURAN ===
        end_time = time.time()  # Catat waktu selesai
        processing_time = end_time - start_time  # Hitung durasi

        # 4. Susun dictionary 'prediction' dengan data baru
        prediction = {
            'class': predicted_class,
            'confidence': f"{confidence:.2%}",
            'processing_time': f"{processing_time:.3f}",  # Format waktu proses
            'metadata': {
                'dimensions': dimensions,
                'size': formatted_size,
            }
        }

        # Konversi gambar ke base64 untuk ditampilkan di HTML
        # Gunakan objek 'image' yang sudah ada
        img_io = io.BytesIO()
        image.save(img_io, 'PNG')
        img_io.seek(0)
        image_data = base64.b64encode(img_io.getvalue()).decode()

    return render_template('index.html', prediction=prediction, image_data=image_data)


if __name__ == '__main__':
    app.run(debug=True, port=params['port'], host='0.0.0.0')