print("[INFO] Import necessary libraries...")

import os
import yaml
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

# If everthing is ok, print the parameters
print("[INFO] Configuration parameters...")
print("* Batch size........................:", params['batchSize'])
print("* Image height......................:", params['imgHeight'])
print("* Image width.......................:", params['imgWidth'])
print("* Number of epochs..................:", params['epochs'])
print("* Train dataset path................:", params['trainDatasetPath'])
print("* Train model path..................:", params['modelPath'])

classification = Classification(params)
classification.execute()
