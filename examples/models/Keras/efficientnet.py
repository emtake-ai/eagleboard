import tensorflow as tf 
from tensorflow.keras import layers, Model

train_dir = "./data/train"
val_dir = "./data/val"

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    image_size=(224,224),
    batch_size=32
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    val_dir,
    image_size=(224,224),
    batch_size=32
)


normalization = layers.Rescaling(1./255)

train_ds = train_ds.map(lambda x, y: (normalization(x), y))
val_ds   = val_ds.map(lambda x, y: (normalization(x), y))

def mbconv_block(inputs, out_channels, expansion=6, kernel_size=3, stride=1, reduction=4):
    in_channels = inputs.shape[-1]

    # ------------------------
    # Expansion (PW)
    # ------------------------
    x = layers.Conv2D(in_channels * expansion, 1, padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("swish")(x)

    # ------------------------
    # Depthwise
    # ------------------------
    x = layers.DepthwiseConv2D(kernel_size, strides=stride, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("swish")(x)

    # ------------------------
    # SE
    # ------------------------
    se = layers.GlobalAveragePooling2D()(x)
    se = layers.Reshape((1, 1, x.shape[-1]))(se)
    se = layers.Dense(x.shape[-1] // reduction, activation="swish")(se)
    se = layers.Dense(x.shape[-1], activation="sigmoid")(se)
    x = layers.Multiply()([x, se])

    # ------------------------
    # Projection (PW)
    # ------------------------
    x = layers.Conv2D(out_channels, 1, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    # ------------------------
    # Residual (조건부)
    # ------------------------
    if stride == 1 and in_channels == out_channels:
        x = layers.Add()([inputs, x])

    return x

def build_efficientnet(input_shape=(224,224,3), num_classes=10):
    inputs = layers.Input(shape=input_shape)

    # Stem
    x = layers.Conv2D(32, 3, strides=2, padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("swish")(x)

    # Blocks (이건 B0 형태 기반 축소 버전)
    x = mbconv_block(x, 16, expansion=1, kernel_size=3, stride=1)
    x = mbconv_block(x, 24, expansion=6, kernel_size=3, stride=2)
    x = mbconv_block(x, 40, expansion=6, kernel_size=5, stride=2)
    x = mbconv_block(x, 80, expansion=6, kernel_size=3, stride=2)

    # Head
    x = layers.Conv2D(1280, 1, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("swish")(x)

    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return Model(inputs, outputs)

model = build_efficientnet()
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"]
)

history = model.fit(
    train_ds,
    epochs=20,
    validation_data=val_ds
)

model.save("efficientnet_custom.keras")