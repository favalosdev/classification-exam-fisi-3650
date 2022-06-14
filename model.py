import tensorflow as tf
import numpy as np

from PIL import Image

class Model:

    def __init__(self):
        self.model = tf.keras.models.load_model('model.h5')
        self.image_width = 256
        self.image_height = 256

    def predict(self, file_path):
        # Load image
        image = Image.open(file_path)

        # Resize it
        image = image.resize((self.image_width, self.image_height))

        # Convert into something our model can understand
        image_array = np.array(image)
        image_array = np.expand_dims(image_array, axis=0)

        predicted_class = np.argmax(self.model.predict(image_array))

        return predicted_class