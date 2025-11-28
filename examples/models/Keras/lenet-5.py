import tensorflow as tf 
from tensorflow.keras import layers, models, datasets 
import numpy as np 

(train_x, train_y), (test_x, test_y) = datasets.mnist.load_data()

# Convert to float32 and normalize
train_x = train_x.astype("float32") / 255.0
test_x = test_x.astype("float32") / 255.0

# Expand dims to add the channel axis → (28, 28, 1)
train_x = tf.expand_dims(train_x, axis=-1)
test_x = tf.expand_dims(test_x, axis=-1)

# ✅ Convert grayscale (1 channel) → RGB (3 channels)
train_x = tf.image.grayscale_to_rgb(train_x)
test_x = tf.image.grayscale_to_rgb(test_x)

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

# ✅ Input layer
inputs = Input(shape=(28, 28, 3))

# ✅ Convolution + Pooling blocks
x = Conv2D(6, kernel_size=(5, 5), strides=1, activation='relu')(inputs)
x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)

x = Conv2D(16, kernel_size=(5, 5), strides=1, activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)

# ✅ Fully-connected layers
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dense(64, activation='relu')(x)
outputs = Dense(10, activation='softmax')(x)

# ✅ Model definition
model = Model(inputs=inputs, outputs=outputs)

# ✅ Optional compile & summary

model.summary()

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])
model.fit(train_x, train_y, epochs=2, batch_size=64)
test_loss, test_acc = model.evaluate(test_x, test_y, verbose=2)
print(f"✅ Test accuracy: {test_acc:.4f}")
model.save("lenet5.keras")