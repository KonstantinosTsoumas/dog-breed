import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    precision_score, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from src.entity.config_entity import EvaluationConfig
from src.utils.auxiliary_functions import save_json
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, label_binarize
from datetime import datetime
import os
from sklearn.metrics import classification_report


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

        # Creating a classification report per class
        report = classification_report(y_true, y_pred)

        # Writing report to tzt
        with open('classification_report.txt', 'w') as file:
            file.write(report)

        # Get precision, recall for each class
        precision = dict()
        recall = dict()
        average_precision = dict()
        for i in range(self.num_classes):
            precision[i], recall[i], _ = precision_recall_curve(y_true_binarized[:, i], predictions[:, i])
            average_precision[i] = average_precision_score(y_true_binarized[:, i], predictions[:, i])

        # Plot precision-recall curve for each class
        for i in range(self.num_classes):
            fig, ax = plt.subplots()
            # Plot the precision-recall curve
            ax.step(recall[i], precision[i], where='post')
            # Set labels, limits and title
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_ylim([0.0, 1.05])
            ax.set_xlim([0.0, 1.0])
            ax.set_title(f'Precision-Recall curve: AP={average_precision[i]:0.2f}')
            self.save_plot(fig=fig, path=self.config.figures, filename=f"classification_report_class_{i}.png")
            # Free up memory
            plt.close(fig)

        return scores, y_true_binarized, predictions

    def save_score(self, scores):
        """
        This method saves the evaluation scores to a JSON file.

        Args:
            scores: dict, A dictionary containing various evaluation metrics.
        returns: -
        """

        save_json(path=Path("scores.json"), data=scores)

    def plot_top_roc_curves(self, scores, y_true_binarized, predictions, top_n=20):
        """
        This method plots the top 20 classes (ROC).

        Args:
            scores: dict, A dictionary containing various evaluation metrics.
            y_true_binarized: array (like), The true binary labels in binary label indicators.
            predictions: array-like, Predictions of the model.
            top_n: int, The top N classes to plot (default = 20).
        returns: -
        """

        avg_precision_scores = scores['average_precision_scores']
        sorted_classes = sorted(avg_precision_scores, key=avg_precision_scores.get, reverse=True)[:top_n]

        # Define the figure and axis object
        fig, ax = plt.subplots(figsize=(15, 10))

        for class_label in sorted_classes:
            class_index = int(class_label.split('_')[-1])
            y_true_binary = y_true_binarized[:, class_index]
            y_score = predictions[:, class_index]

            fpr, tpr, _ = roc_curve(y_true_binary, y_score)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'Class {class_index} (AUC = {roc_auc:.2f})')

        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'Top {top_n} ROC Curves')
        ax.legend(loc="lower right")

        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"roc_curve_{current_time}.png"
        self.save_plot(fig=fig, path=self.config.figures, filename=filename)
    @staticmethod
    def save_plot(fig, path: str, filename: str):
        """
        This function saves a plot to the specified path.

        Args:
            fig : matplotlib.figure.Figure, the figure object to be saved.
            path : str, the directory path where the plot should be saved.
            filename : str, the filename for the saved plot.
        returns: -
        """
        full_path = os.path.join(path, filename)
        fig.savefig(full_path)