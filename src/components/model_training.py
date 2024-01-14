from src.entity.config_entity import TrainingConfig
import tensorflow as tf
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def get_base_model(self):
        """
        This function loads the base model from the specified path in the configuration.
        """
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )

    def create_tf_dataset(self):
        """
        This function creates and returns TensorFlow datasets for training and validation.
        returns: tuple, containing both train and validation sets.
        """

        # Read the CSV
        labels_df = pd.read_csv(Path(self.config.artifacts, 'labels.csv'))

        # The 'id' column contains the image filename without extension
        labels_df['filename'] = labels_df['id'].apply(lambda x: str(Path(self.config.training_data, x + '.jpg')))

        # Convert breed names to numerical labels for tensorflow
        label_encoder = LabelEncoder()
        labels_df['breed_label'] = label_encoder.fit_transform(labels_df['breed'])

        # Process the images, add 120 tensors.
        def process_path(file_path, label):
            """
            This function processes a given image file path into a tensor and encodes its label.
            returns: tuple, containing the image tensor and its encoded label.

            """
            img = tf.io.read_file(file_path)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, self.config.params_image_size[:-1])
            img = tf.keras.applications.resnet_v2.preprocess_input(img)
            label = tf.one_hot(label, depth=120)
            return img, label

        # Create a dataset
        list_ds = tf.data.Dataset.from_tensor_slices((labels_df['filename'].values, labels_df['breed_label'].values))
        dataset = list_ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)

        # Shuffle and split (90/10)
        dataset = dataset.shuffle(buffer_size=len(labels_df))
        train_size = int(0.9 * len(labels_df))
        train_ds = dataset.take(train_size)
        val_ds = dataset.skip(train_size)

        # Batch it
        train_ds = train_ds.batch(self.config.params_batch_size).prefetch(tf.data.AUTOTUNE)
        val_ds = val_ds.batch(self.config.params_batch_size).prefetch(tf.data.AUTOTUNE)

        return train_ds, val_ds

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        """
        This function saves the trained model to the specified path.

        Args:
            path : Path, the path where the model should (is) be saved.
             model : tf.keras.Model, the trained TensorFlow model to be saved.
        """
        model.save(path)

    def train(self, callback_list: list):
        """
        This function trains the model using the training and validation datasets.
        Args:
            callback_list : list, the list of callbacks to be used during training.
        """

        train_ds, val_ds = self.create_tf_dataset()

        self.model.fit(
            train_ds,
            epochs=self.config.params_epochs,
            validation_data=val_ds,
            callbacks=callback_list
        )

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )