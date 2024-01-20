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
