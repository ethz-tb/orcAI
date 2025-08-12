import os
from importlib import metadata

__version__ = metadata.version("orcai")

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
