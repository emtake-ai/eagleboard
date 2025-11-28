import tensorflow as tf
from tensorflow.keras import layers, models

# ------------------------------
# Dataset (same as before)
# ------------------------------
train_dir = "data/train"
val_dir = "data/val"

train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    labels="inferred",
    label_mode="int",
    image_size=(224, 224),
    batch_size=32,
    shuffle=True
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    labels="inferred",
    label_mode="int",
    image_size=(224, 224),
    batch_size=128,
    shuffle=False
)

AUTOTUNE = tf.data.AUTOTUNE
normalization_layer = tf.keras.layers.Rescaling(1./255)

train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y),
                        num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y),
                    num_parallel_calls=AUTOTUNE)

train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# ------------------------------
# 🔹 AlexNet Functional Model
# ------------------------------
inputs = layers.Input(shape=(224, 224, 3))

# 1st conv block
x = layers.Conv2D(96, (11, 11), strides=4, padding="valid")(inputs)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.MaxPool2D((3, 3), strides=2)(x)

# 2nd conv block
x = layers.Conv2D(256, (5, 5), padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.MaxPool2D((3, 3), strides=2)(x)

# 3rd–5th conv blocks
x = layers.Conv2D(384, (3, 3), padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)

x = layers.Conv2D(384, (3, 3), padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)

x = layers.Conv2D(256, (3, 3), padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.MaxPool2D((3, 3), strides=2)(x)

# FC layers
x = layers.Flatten()(x)
x = layers.Dense(4096, activation='relu')(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(4096, activation='relu')(x)
x = layers.Dropout(0.5)(x)

# Output
outputs = layers.Dense(2, activation='softmax')(x)

# Build model
model = models.Model(inputs=inputs, outputs=outputs, name="AlexNet")

# ------------------------------
# Compile & Train
# ------------------------------
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=['accuracy'])

model.summary()
model.fit(train_ds, validation_data=val_ds, epochs=10)

# ------------------------------
# Save for Synabro
# ------------------------------
model.save("alexnet.keras")
