import os
from pathlib import Path
import logging

# Generate log messages
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

# Define project name
project_name = "dog-breed"

# Configure paths
list_of_files = [
    ".github/workflows/.gitkeep",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/constants/__init__.py",
    "config/config.yaml",
    "dvc.yaml",
    "params.yaml",
    "requirements.txt",
    "setup.py",
    "research/trials.ipynb",
    "templates/index.html"
]

for path in list_of_files:
    path = Path(path)
    filedir, filename = os.path.split(path)


    # Create directory if non-existing
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory; {filedir} for the file: {filename}")

    # Create an empty default file if non-existing
    if (not os.path.exists(path)) or (os.path.getsize(path) == 0):
        with open(path, "w") as f:
            pass
            logging.info(f"Creating empty file: {path}")


    else:
        logging.info(f"{filename} is already exists")