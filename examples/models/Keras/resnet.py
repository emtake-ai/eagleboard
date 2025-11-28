import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, Activation,
    Add, MaxPooling2D, GlobalAveragePooling2D, Dense
)
from tensorflow.keras.models import Model

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

num_classes = 2   # ✅ set for binary classification

# ---------------------------------------------------
# 🔹 Residual Blocks
# ---------------------------------------------------
def identity_block(x, filters):
    f1, f2, f3 = filters
    shortcut = x

    x = Conv2D(f1, (1, 1), padding='valid')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(f2, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(f3, (1, 1), padding='valid')(x)
    x = BatchNormalization()(x)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x


def conv_block(x, filters, stride=2):
    f1, f2, f3 = filters
    shortcut = x

    # Main path
    x = Conv2D(f1, (1, 1), strides=stride, padding='valid')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(f2, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(f3, (1, 1), padding='valid')(x)
    x = BatchNormalization()(x)

    # Shortcut path
    shortcut = Conv2D(f3, (1, 1), strides=stride, padding='valid')(shortcut)
    shortcut = BatchNormalization()(shortcut)

    # Merge
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x


# ---------------------------------------------------
# 🔹 ResNet-50 Architecture
# ---------------------------------------------------
def ResNet50(input_shape=(224, 224, 3), classes=2):
    inputs = Input(shape=input_shape)

    # Stage 1
    x = Conv2D(64, (7, 7), strides=2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=2, padding='same')(x)

    # Stage 2
    x = conv_block(x, [64, 64, 256], stride=1)
    x = identity_block(x, [64, 64, 256])
    x = identity_block(x, [64, 64, 256])

    # Stage 3
    x = conv_block(x, [128, 128, 512])
    x = identity_block(x, [128, 128, 512])
    x = identity_block(x, [128, 128, 512])
    x = identity_block(x, [128, 128, 512])

    # Stage 4
    x = conv_block(x, [256, 256, 1024])
    for _ in range(5):
        x = identity_block(x, [256, 256, 1024])

    # Stage 5
    x = conv_block(x, [512, 512, 2048])
    x = identity_block(x, [512, 512, 2048])
    x = identity_block(x, [512, 512, 2048])

    # Output
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(classes, activation='softmax')(x)

    return Model(inputs, outputs, name='ResNet50')


# ---------------------------------------------------
# 🔹 Build, compile, and train
# ---------------------------------------------------
model = ResNet50(input_shape=(224, 224, 3), classes=num_classes)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# ✅ Train
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

# ✅ Save
model.save("resnet50.keras")
print("[INFO] Training complete — model saved as resnet50.keras")
