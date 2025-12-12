Data Preprocessing (PyTorch & TensorFlow Comparison)


Data preprocessing is a critical step in deep learning pipelines.
It ensures that input data is transformed into a consistent and normalized format before being fed into the model.
Typical preprocessing steps include resizing, normalization, data type conversion, and optional data augmentation.
This document compares preprocessing approaches in PyTorch and TensorFlow (Keras).


1. PyTorch Data Preprocessing
In PyTorch, preprocessing is commonly handled using torchvision.transforms.
Transforms are applied to each sample when it is loaded from the Dataset.

Example preprocessing steps in PyTorch:

from torchvision import transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])
])


2. TensorFlow (Keras) Data Preprocessing

In TensorFlow, preprocessing is typically implemented using the tf.data pipeline.
Preprocessing functions are applied to dataset elements using the map function.

Example preprocessing steps in TensorFlow (Keras):

import tensorflow as tf

def preprocess(image, label):
    image = tf.image.resize(image, (224, 224))
    image = tf.cast(image, tf.float32) / 255.0
    image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    return image, label

