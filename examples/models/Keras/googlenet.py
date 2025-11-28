import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Dropout,
    Dense, Concatenate, GlobalAveragePooling2D
)
from tensorflow.keras.models import Model


# ---------------------------------------------------
# 🔹 Inception block definition
# ---------------------------------------------------
def inception_block(x, f1, f3r, f3, f5r, f5, fp):
    path1 = Conv2D(f1, (1, 1), activation='relu', padding='same')(x)

    path2 = Conv2D(f3r, (1, 1), activation='relu', padding='same')(x)
    path2 = Conv2D(f3, (3, 3), activation='relu', padding='same')(path2)

    path3 = Conv2D(f5r, (1, 1), activation='relu', padding='same')(x)
    path3 = Conv2D(f5, (5, 5), activation='relu', padding='same')(path3)

    path4 = MaxPooling2D((3, 3), strides=1, padding='same')(x)
    path4 = Conv2D(fp, (1, 1), activation='relu', padding='same')(path4)

    return Concatenate(axis=-1)([path1, path2, path3, path4])


# ---------------------------------------------------
# 🔹 GoogLeNet (Inception v1)
# ---------------------------------------------------
def GoogLeNet(input_shape=(224, 224, 3), classes=10):
    inputs = Input(shape=input_shape)

    x = Conv2D(64, (7, 7), strides=2, padding='same', activation='relu')(inputs)
    x = MaxPooling2D((3, 3), strides=2, padding='same')(x)

    x = Conv2D(64, (1, 1), activation='relu', padding='same')(x)
    x = Conv2D(192, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((3, 3), strides=2, padding='same')(x)

    x = inception_block(x, 64, 96, 128, 16, 32, 32)
    x = inception_block(x, 128, 128, 192, 32, 96, 64)
    x = MaxPooling2D((3, 3), strides=2, padding='same')(x)

    x = inception_block(x, 192, 96, 208, 16, 48, 64)
    x = inception_block(x, 160, 112, 224, 24, 64, 64)
    x = inception_block(x, 128, 128, 256, 24, 64, 64)
    x = inception_block(x, 112, 144, 288, 32, 64, 64)
    x = inception_block(x, 256, 160, 320, 32, 128, 128)
    x = MaxPooling2D((3, 3), strides=2, padding='same')(x)

    x = inception_block(x, 256, 160, 320, 32, 128, 128)
    x = inception_block(x, 384, 192, 384, 48, 128, 128)

    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.4)(x)
    outputs = Dense(classes, activation='softmax')(x)

    return Model(inputs, outputs, name='GoogLeNet')


# ---------------------------------------------------
# 🔹 Dataset
# ---------------------------------------------------
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

# ---------------------------------------------------
# 🔹 Build, compile, and train
# ---------------------------------------------------
num_classes = 2
model = GoogLeNet(input_shape=(224, 224, 3), classes=num_classes)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# ✅ Train the model
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

# ✅ Save
model.save("googlenet.keras")
print("[INFO] Training complete — model saved as googlenet.keras")
