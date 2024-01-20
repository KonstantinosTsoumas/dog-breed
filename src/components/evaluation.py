import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    precision_score, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from src.entity.config_entity import EvaluationConfig
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from pathlib import Path
from src.utils.auxiliary_functions import save_json
from sklearn.preprocessing import LabelEncoder, label_binarize

class Evaluation:

    def __init__(self, config: EvaluationConfig):
        """
        This method initializes the Evaluation class given the configuration.

        Args:
            config (EvaluationConfig): An instance of EvaluationConfig class.
        """
        self.num_classes = None
        self.config = config

    def create_tf_dataset(self):
        """
        Creates and returns a TensorFlow dataset for evaluation.
        """
        # Read the CSV
        labels_df = pd.read_csv(Path(self.config.artifacts, 'labels.csv'))

        # Prepare the file paths and the labels
        labels_df['filename'] = labels_df['id'].apply(lambda x: str(Path(self.config.training_data, x + '.jpg')))
        label_encoder = LabelEncoder()
        labels_df['label'] = label_encoder.fit_transform(labels_df['breed'])

        # Determine the number of classes
        self.num_classes = len(label_encoder.classes_)

        def process_path(file_path, label):
            img = tf.io.read_file(file_path)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, self.config.params_image_size[:-1])
            img = tf.keras.applications.resnet_v2.preprocess_input(img)
            label = tf.one_hot(label, depth=len(label_encoder.classes_))
            return img, label

        # Create a dataset
        list_ds = tf.data.Dataset.from_tensor_slices((labels_df['filename'].values, labels_df['label'].values))
        dataset = list_ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)

        # Batch it
        dataset = dataset.batch(self.config.params_batch_size).prefetch(tf.data.AUTOTUNE)

        return dataset

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        """
        This static method loads a Keras model from the specified path.

        Args:
            path: path, The path to the saved model.

        Returns:
            Loaded Keras model.
        """

        return tf.keras.models.load_model(path)

    def evaluation(self):
        """
        This method evaluates the model using the validation set.

        Specifically, it calculates and stores various evaluation metrics including loss, accuracy, precision, ROC AUC, and average precision.
        """

        model = self.load_model(self.config.path_of_model)
        eval_ds = self.create_tf_dataset()

        predictions = model.predict(eval_ds)
        y_pred = np.argmax(predictions, axis=1)
        y_true = np.concatenate([y for x, y in eval_ds], axis=0)
        y_true = np.argmax(y_true, axis=1)

        self.score = model.evaluate(eval_ds)
        scores = {"loss": self.score[0], "accuracy": self.score[1]}

        precision = precision_score(y_true, y_pred, average=None)
        scores['precision'] = precision.tolist()

        # Binarize the labels for multiclass ROC AUC and average precision
        y_true_binarized = label_binarize(y_true, classes=range(self.num_classes))

        roc_auc_scores = {}
        average_precisions = {}
        for i in range(self.num_classes):
            # Calculate ROC AUC for each class (OvR strategy for multi class)
            fpr, tpr, _ = roc_curve(y_true_binarized[:, i], predictions[:, i])
            roc_auc_scores[f"roc_auc_class_{i}"] = auc(fpr, tpr)

            # Calculate average precision for each class
            if np.sum(y_true_binarized[:, i]) > 0:
                average_precisions[f"average_precision_class_{i}"] = average_precision_score(y_true_binarized[:, i], predictions[:, i])
            else:
                average_precisions[f"average_precision_class_{i}"] = float('nan')

        scores['roc_auc_scores'] = roc_auc_scores
        scores['average_precision_scores'] = average_precisions

        return scores
        #self.save_score(scores)

    def save_score(self, scores):
        """
        This method saves the evaluation scores to a JSON file.

        Args:
            scores: dict, A dictionary containing various evaluation metrics.
        returns: -
        """

        save_json(path=Path("scores.json"), data=scores)

