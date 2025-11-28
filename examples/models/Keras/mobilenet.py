import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os

# ============================================================
# 📁 Dataset Root
# ============================================================
dataset_root = "/home/soo/Work/Data/Thermal_Kevin"

train_images = dataset_root + "/images/train"
train_labels = dataset_root + "/labels/train"

val_images = dataset_root + "/images/val"
val_labels = dataset_root + "/labels/val"

# ============================================================
# 📌 Model Parameters
# ============================================================
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 2   # 필요에 따라 수정

# ============================================================
# 1) Image Load (.jpg)
# ============================================================
def load_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    return image

# ============================================================
# 2) Resize + Normalize
# ============================================================
def preprocess_image(image):
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return image

# ============================================================
# 3) Label Load (txt: "0", "1" ...)
# ============================================================
def load_label(path):
    txt = tf.io.read_file(path)
    label = tf.strings.to_number(txt, out_type=tf.int32)
    return label

# ============================================================
# 4) Combine Image + Label
# ============================================================
def load_and_preprocess(image_path, label_path):
    image = load_image(image_path)
    image = preprocess_image(image)
    label = load_label(label_path)
    return image, label


# ============================================================
# 📌 Dataset Builder
# ============================================================
import re

# 숫자부분 추출 함수
def extract_number(filename):
    # captured_123.png → 123
    nums = re.findall(r'\d+', filename)
    return int(nums[-1]) if nums else -1


def build_dataset(image_dir, label_dir):

    # 이미지 목록
    image_files = [
        f for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    # 라벨 목록 (_cls.txt 만 사용)
    label_files = [
        f for f in os.listdir(label_dir)
        if f.endswith("_cls.txt")
    ]

    # 자연 정렬
    image_files = sorted(image_files, key=extract_number)
    label_files = sorted(label_files, key=extract_number)

    # 전체 경로로 변환
    image_paths = [os.path.join(image_dir, f) for f in image_files]
    label_paths = [os.path.join(label_dir, f) for f in label_files]

    # 개수 검사
    assert len(image_paths) == len(label_paths), \
        f"Mismatch: {len(image_paths)} images vs {len(label_paths)} labels"

    # 이름 매칭 검사
    for img, lab in zip(image_paths, label_paths):
        img_id = os.path.splitext(os.path.basename(img))[0]          # captured_1
        lab_id = os.path.basename(lab).replace("_cls.txt", "")       # captured_1
        assert img_id == lab_id, f"Name mismatch: {img_id} vs {lab_id}"

    # TF dataset 구성
    ds = tf.data.Dataset.from_tensor_slices((image_paths, label_paths))
    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return ds


# ============================================================
# 📌 Train / Val Dataset Load
# ============================================================
train_ds = build_dataset(train_images, train_labels)
val_ds   = build_dataset(val_images, val_labels)

print("✔ Loaded train_ds and val_ds")
print(f"Train batches: {len(train_ds)}, Val batches: {len(val_ds)}")


# ============================================================
# 🔥  MobileNetV3 Small Implementation
# ============================================================

def h_sigmoid(x):
    return layers.ReLU(max_value=6)(x + 3) / 6.0

def SEBlock(x, reduction=4):
    filters = x.shape[-1]

    se = layers.GlobalAveragePooling2D()(x)
    se = layers.Reshape((1, 1, filters))(se)

    se = layers.Conv2D(filters // reduction, 1, activation='relu')(se)
    se = layers.Conv2D(filters, 1)(se)      # no activation
    se = tf.nn.relu(se)
                    # replace sigmoid

    return layers.Multiply()([x, se])

def MBConv(x, out_channels, kernel, stride, expand_ratio, use_se, activation='swish'):
    in_channels = x.shape[-1]
    hidden_dim = in_channels * expand_ratio

    # Expand
    if expand_ratio != 1:
        x_expanded = layers.Conv2D(hidden_dim, 1, padding='same', use_bias=False)(x)
        x_expanded = layers.BatchNormalization()(x_expanded)
        x_expanded = layers.Activation(activation)(x_expanded)
    else:
        x_expanded = x

    # Depthwise
    x_dw = layers.DepthwiseConv2D(kernel, strides=stride, padding='same', use_bias=False)(x_expanded)
    x_dw = layers.BatchNormalization()(x_dw)
    x_dw = layers.Activation(activation)(x_dw)

    # SE
    x_se = SEBlock(x_dw) if use_se else x_dw

    # Project
    x_pj = layers.Conv2D(out_channels, 1, padding='same', use_bias=False)(x_se)
    x_pj = layers.BatchNormalization()(x_pj)

    # Residual
    if stride == 1 and in_channels == out_channels:
        return layers.Add()([x, x_pj])
    return x_pj


def MobileNetV3_Small(input_shape=(224, 224, 3), num_classes=2):
    inputs = layers.Input(shape=input_shape)

    # Stem
    x = layers.Conv2D(16, 3, strides=2, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)

    # Stages
    x = MBConv(x, 16, 3, 2, 1, True)
    x = MBConv(x, 24, 3, 2, 4, False)
    x = MBConv(x, 24, 3, 1, 3, False)
    x = MBConv(x, 40, 5, 2, 3, True)
    x = MBConv(x, 40, 5, 1, 3, True)
    x = MBConv(x, 40, 5, 1, 3, True)
    x = MBConv(x, 48, 5, 1, 3, True)
    x = MBConv(x, 48, 5, 1, 3, True)
    x = MBConv(x, 96, 5, 2, 6, True)
    x = MBConv(x, 96, 5, 1, 6, True)
    x = MBConv(x, 96, 5, 1, 6, True)

    # Final Layers
    x = layers.Conv2D(576, 1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1024, activation='swish')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return models.Model(inputs, outputs, name="MobileNetV3_Small")


# ============================================================
# 🚀 Build Model
# ============================================================
model = MobileNetV3_Small(input_shape=(224, 224, 3), num_classes=NUM_CLASSES)
model.summary()

# ============================================================
# 🚀 Compile
# ============================================================
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

# ============================================================
# 🚀 Fit
# ============================================================
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=2
)

# ============================================================
# 🚀 Evaluate
# ============================================================
loss, acc = model.evaluate(val_ds)

model.save("mobilenetv3.keras")
print("\n==============================")
print("Validation Loss:", loss)
print("Validation Acc :", acc)
print("==============================")
