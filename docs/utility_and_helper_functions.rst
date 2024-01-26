Utility and Helper Functions
============================

This document provides detailed information about various utility and helper functions used in the project. These functions facilitate operations such as file handling, data processing, and image manipulation.

Functions
---------

1. read_yaml
-----------

.. code-block:: python

   @ensure_annotations
   def read_yaml(path_to_yaml: Path) -> ConfigBox:
       """
       Reads a YAML file and returns its contents as a ConfigBox object.

       :param path_to_yaml: Path to the YAML file.
       :type path_to_yaml: Path
       :return: Contents of the YAML file as a ConfigBox.
       :rtype: ConfigBox
       :raises ValueError: If the YAML file is empty.
       :raises Exception: For other exceptions during file reading.
       """

2. create_directories
---------------------

.. code-block:: python

   @ensure_annotations
   def create_directories(path_to_directories: list, verbose=True):
       """
       Creates a list of directories.

       :param path_to_directories: List of directory paths to be created.
       :type path_to_directories: list
       :param verbose: If True, logs the creation of each directory. Defaults to True.
       :type verbose: bool
       """

3. save_json
-----------

.. code-block:: python

   @ensure_annotations
   def save_json(path: Path, data: dict):
       """
       Saves a dictionary as a JSON file.

       :param path: Path to save the JSON file.
       :type path: Path
       :param data: Data to be saved in JSON format.
       :type data: dict
       """

4. load_json
-----------

.. code-block:: python

   @ensure_annotations
   def load_json(path: Path) -> ConfigBox:
       """
       Loads a JSON file and returns its contents as a ConfigBox.

       :param path: Path to the JSON file.
       :type path: Path
       :return: Contents of the JSON file as a ConfigBox.
       :rtype: ConfigBox
       """

5. save_bin
----------

.. code-block:: python

   @ensure_annotations
   def save_bin(data: Any, path: Path):
       """
       Saves data in binary format.

       :param data: Data to be saved.
       :type data: Any
       :param path: Path to save the binary file.
       :type path: Path
       """

6. load_bin
----------

.. code-block:: python

   @ensure_annotations
   def load_bin(path: Path) -> Any:
       """
       Loads binary data from a file.

       :param path: Path to the binary file.
       :type path: Path
       :return: Data loaded from the binary file.
       :rtype: Any
       """

7. get_size
----------

.. code-block:: python

   @ensure_annotations
   def get_size(path: Path) -> str:
       """
       Returns the size of a file in kilobytes.

       :param path: Path of the file.
       :type path: Path
       :return: Size of the file in KB.
       :rtype: str
       """

8. decode_image
--------------

.. code-block:: python

   def decode_image(img_string, file_name):
       """
       Decodes a base64 encoded image string and saves it as a file.

       :param img_string: Base64 encoded string of the image.
       :type img_string: str
       :param file_name: Name of the file to save the decoded image.
       :type file_name: str
       """

9. encode_image_into_base64
---------------------------

.. code-block:: python

   def encode_image_into_base64(cropped_image_path):
       """
       Encodes an image file into a base64 string.

       :param cropped_image_path: Path to the image file to be encoded.
       :type cropped_image_path: str
       :return: Base64 encoded string of the image.
       :rtype: str
       """

