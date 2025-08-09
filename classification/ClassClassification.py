# Import necessary libraries
import matplotlib
import tensorflow as tf
import matplotlib.pyplot as plt
import os

# Set the backend for matplotlib to avoid GUI issues in headless environments
matplotlib.use('Agg')

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# Define the Classification class
class Classification:

    # Constructor
    def __init__(self, parameters):

        # Initialize parameters
        self.parameters = parameters
        self.path = parameters['trainDatasetPath']
        self.modelPath = parameters['modelPath']
        self.batchsz = parameters['batchSize']
        self.epochs = parameters['epochs']
        self.imgWidth = parameters['imgWidth']
        self.imgHeight = parameters['imgHeight']
        self.modelPath = parameters['modelPath']
        self.modelName = parameters['modelName']

    # Check if model is already trained?
    def isModelTrained(self):
        print("[INFO] Checking trained model...")

        if not os.path.exists(self.modelPath):
            print("[INFO] Model not found at {}, training required.".format(self.modelPath))

            # If model is not trained, train the model
            model = self.execute()
            return model

        else:
            print("[INFO] Model found at {}, loading...".format(self.modelPath))

            # If model is trained, load the model
            model = tf.keras.models.load_model(self.modelPath)
            return model

    # Function to load the dataset
    def loadDataset(self):
        print("[INFO] Loading dataset...")

        # Load the training datasets from the directory
        trainDataset = tf.keras.utils.image_dataset_from_directory(
            self.path,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(self.imgHeight, self.imgWidth),
            batch_size=self.batchsz,
        )

        # Load the validation dataset from the directory
        validDataset = tf.keras.utils.image_dataset_from_directory(
            self.path,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(self.imgHeight, self.imgWidth),
            batch_size=self.batchsz,
        )

        # Get the class names and number of classes
        classNames = trainDataset.class_names
        numClasses = len(classNames)

        print("[INFO] Number of images: {}".format(len(trainDataset)))
        print("[INFO] Class names: {}".format(classNames))

        # Caching, prefetching and shuffling the datasets for performance
        autotune = tf.data.AUTOTUNE
        trainDataset.cache().shuffle(1000).prefetch(buffer_size=autotune)
        validDataset.cache().prefetch(buffer_size=autotune)

        # Return variables
        return trainDataset, validDataset, classNames, numClasses

    # Function to build the model using MobileNetV3 (Transfer Learning)
    def buildMobileNetV3Model(self, numClasses):
        """
        Membangun model menggunakan MobileNetV3 dengan transfer learning.
        Versi ini sudah dikoreksi dengan preprocessing dan augmentasi yang benar.
        """
        print(f"[INFO] Building model using {self.modelName} (Transfer Learning)...")

        # LANGKAH 1: Definisikan input dan lapisan augmentasi
        inputs = tf.keras.Input(shape=(self.imgHeight, self.imgWidth, 3))

        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.2),
            tf.keras.layers.RandomZoom(0.1),
        ], name='data_augmentation')

        x = data_augmentation(inputs)

        # LANGKAH 2: Terapkan preprocessing yang BENAR untuk MobileNet
        # Ini adalah koreksi paling kritis. Mengubah skala piksel ke [-1, 1].
        x = tf.keras.applications.mobilenet_v3.preprocess_input(x)

        # LANGKAH 3: Pilih dan muat model dasar (base model)
        if self.modelName == 'MobileNetV3Small':
            base_model = tf.keras.applications.MobileNetV3Small(
                input_shape=(self.imgHeight, self.imgWidth, 3),
                include_top=False,
                weights='imagenet'
            )
        else:  # Asumsikan 'MobileNetV3Large' atau lainnya
            base_model = tf.keras.applications.MobileNetV3Large(
                input_shape=(self.imgHeight, self.imgWidth, 3),
                include_top=False,
                weights='imagenet'
            )

        # Bekukan bobot dari model dasar agar tidak ikut terlatih
        base_model.trainable = False

        # LANGKAH 4: Sambungkan ke model dasar dan buat kepala klasifikasi
        x = base_model(x, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = tf.keras.layers.Dense(numClasses, activation='softmax')(x)

        # LANGKAH 5: Buat model final
        model = tf.keras.Model(inputs, outputs)
        return model

    # Function to build the model
    def buildCustomModel(self, imgWidth, imgHeight, numClasses):
        print("[INFO] Building model...")

        # Create the data augmentation layer
        dataAug = tf.keras.Sequential([
            tf.keras.layers.RandomFlip('horizontal',
                                       input_shape=(imgHeight, imgWidth, 3)),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1)
        ])

        # Build the model
        model = tf.keras.Sequential([
            dataAug,
            tf.keras.layers.Rescaling(1. / 255),
            tf.keras.layers.Conv2D(16,
                                   3,
                                   padding='same',
                                   activation='relu'),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Conv2D(32,
                                   3,
                                   padding='same',
                                   activation='relu'),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Conv2D(64,
                                   3,
                                   padding='same',
                                   activation='relu'),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128,
                                  activation='relu'),
            tf.keras.layers.Dense(numClasses,
                                  activation='softmax')
        ])

        # Return the model
        return model

    def buildModel(self, numClasses):
        if self.modelName == 'CustomModel':
            return self.buildCustomModel(numClasses)
        else:
            return self.buildMobileNetV3Model(numClasses)

    # Function to compile the model
    def compileModel(self, model):
        print("[INFO] Compiling model...")

        # Compile the model with the Adam optimizer and sparse categorical crossentropy loss
        model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['acc']
        )

        # Print the model summary
        model.summary()

        # Return the compiled model
        return model

    # Function to train the model
    def trainModel(self, model, trainDataset, validDataset):
        print("[INFO] Training model...")

        # Train the model with the training dataset and validate it using the validation dataset
        history = model.fit(
            trainDataset,
            validation_data=validDataset,
            epochs=self.epochs,
        )

        # Save the model to a file
        model.save(self.modelPath)

        # return the training history
        return history

    # Execute the training process
    def execute(self):
        print("[INFO] Checking for a pre-trained model...")
        if os.path.exists(self.modelPath):
            print(f"[INFO] Model found at {self.modelPath}, loading...")
            model = tf.keras.models.load_model(self.modelPath)
        else:
            print(f"[INFO] Model not found at {self.modelPath}, starting a new training process.")
            model = self._train_new_model()

        print("[INFO] Model is ready for use.")
        return model

    # Private method for the actual training pipeline
    def _train_new_model(self):
        trainDataset, validDataset, _, numClasses = self.loadDataset()
        model = self.buildModel(numClasses)
        model = self.compileModel(model)
        _ = self.trainModel(model, trainDataset, validDataset)
        return model
