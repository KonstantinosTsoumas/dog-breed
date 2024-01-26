Configuration Guide
===================

This section provides details on the configuration files `config.yaml` and `params.yaml` used in the project.

config.yaml
-----------

.. code-block:: yaml

   artifacts_root: artifacts

   data_ingestion:
       root_dir: artifacts/data_ingestion
       source_url: "https://github.com/KonstantinosTsoumas/dog-breed/tree/main/artifacts"
       local_source_file: artifacts/train.zip
       local_data_file: artifacts/train.zip
       unzip_dir: artifacts/data_ingestion/train

   prepare_base_model:
       root_dir: artifacts/prepare_base_model
       base_model_path: artifacts/prepare_base_model/base_model.h5
       updated_base_model_path: artifacts/prepare_base_model/base_model_updated.h5

   prepare_callbacks:
       root_dir: artifacts/prepare_callbacks
       tensorboard_root_log_dir: artifacts/prepare_callbacks/tensorboard_log_dir
       checkpoint_model_filepath: artifacts/prepare_callbacks/checkpoint_dir/model.h5

   training:
       root_dir: artifacts/data_ingestion
       trained_model_path: artifacts/training/model.h5

params.yaml
-----------

.. code-block:: yaml

   AUGMENTATION: True
   IMAGE_SIZE: [224, 224, 3]  # ResNet101V2
   BATCH_SIZE: 32
   INCLUDE_TOP: False
   EPOCHS: 10
   CLASSES: 120
   WEIGHTS: imagenet
   LEARNING_RATE: 0.01
   FREEZE_ALL: True
   FREEZE_TILL: 2
