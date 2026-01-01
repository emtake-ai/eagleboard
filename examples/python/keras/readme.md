## Deep Learning Training Workflow (Keras-Only Version)

This document describes the complete workflow for training a deep learning model using TensorFlow Keras only.  
The workflow covers the entire process from dataset loading to model training and saving the trained model.  

### Training Workflow

The training process follows the steps below.  

```text
1. Dataset Loader
2. Data Preprocessing
3. Deep Learning Modeling
4. Compiler Setting
5. Training Setting
6. Start Training
```

### 1. Dataset Loader  

Keras provides a convenient API called image_dataset_from_directory to load image datasets from a directory structure.  

Example usage:  

```text
import tensorflow as tf
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset/train",
    image_size=(224, 224),
    batch_size=32
)
```

This function automatically infers labels from subdirectory names and returns a tf.data.Dataset object suitable for training.  


### 2. Data Preprocessing

Data preprocessing is applied before feeding data into the model.  
In Keras, preprocessing steps can be added to the dataset pipeline using the map function.  

Example preprocessing logic:   

```text
def preprocess(image, label):
    image = tf.image.resize(image, (224, 224))
    image = tf.cast(image, tf.float32) / 255.0
    return image, label
    train_ds = train_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
```


Preprocessing typically includes resizing, normalization, data type conversion, and optional augmentation.  
The same preprocessing logic must be used during inference.  


### 3. Deep Learning Modeling

In this step, the model architecture is defined.   
Below is an example of a simple CNN-based classification model implemented using Keras Sequential API.  

```text
from tensorflow.keras import layers, models
model = models.Sequential([
    layers.Conv2D(16, (3, 3), activation="relu", padding="same", input_shape=(224, 224, 3)),
    layers.MaxPooling2D(4),
    layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D(4),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax")
])

```

This model consists of convolution, pooling, and fully connected layers suitable for basic image classification tasks.  


### 4. Compiler Setting
Before training starts, training-related parameters must be configured using the compile function.  

Example configuration:  

```text
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
```
This step defines how the model weights will be updated during training.  


### 5. Training Setting

Training parameters such as the number of epochs are defined separately.  
epochs = 5 :   
Epoch specifies how many times the entire dataset is iterated during training.  


### 6. Start Training

Once all settings are complete, training can be started.  

```text
print("Training Started...")
history = model.fit(train_ds, epochs=epochs)
print("Training Completed!")

After training finishes, the trained model is saved using the Keras native format.
model.save("model_name.keras")
```

#### Output
After training is completed, the trained model is saved as:  
```text
model_name.keras
```
The .keras file contains the model architecture, trained weights, and optimizer state.  
This file can be reloaded for further training or converted to ONNX for deployment on other platforms such as NPU-based systems.  


### Summary  
This document describes a Keras-only deep learning training workflow.  
The training output is saved in the native .keras format and can be used in downstream stages such as model conversion and NPU inference.  