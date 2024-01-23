import os
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


class PredictionPipeline:

    def __init__(self, filename):
        self.filename = filename
        self.model = load_model(os.path.join("artifacts", "training", "model.h5"))
        self.label_encoder = joblib.load(os.path.join("artifacts", "label_encoder.pkl"))


    def predict(self) -> dict:
        """
        This method predicts the class of an image. In this context, the dog breed of a given image.

        Returns:
            A list containing a dictionary with the prediction result.
        """

        # Load and preprocess the image.
        test_image = image.load_img(self.filename, target_size=(224, 224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)

        # Predict the class of a given image.
        result = np.argmax(self.model.predict(test_image), axis=1)

        # Decode the numerical prediction to the corresponding dog breed name.
        prediction = self.label_encoder.inverse_transform(result)[0]

        # Return the formatted dict with predictions
        return [{"image": prediction}]
