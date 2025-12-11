📌 딥러닝 학습 전체 워크플로우 (Keras 단독 버전)
이 문서는 TensorFlow Keras 기반으로 딥러닝 모델을 학습할 때 필요한 전체 절차를 정리한 문서입니다.
전체 과정은 다음 순서대로 진행됩니다.
Dataset Loader
Data Preprocessing
Deep Learning Modeling
Compiler Setting
Training Setting
Start Training
아래는 Keras 단독으로 구현한 전체 코드 예시입니다.
1. Dataset Loader (데이터셋 로더)
Keras는 image_dataset_from_directory() API를 통해 매우 간단하게 데이터셋을 불러올 수 있습니다.
이미지 디렉토리는 다음과 같은 구조를 가정합니다:
dataset/
 └── train/
      ├── class1/
      ├── class2/
      └── ...
🔥 Keras 데이터셋 로더 코드
import tensorflow as tf

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset/train",
    image_size=(224, 224),
    batch_size=32
)
2. Data Preprocessing (데이터 전처리)
Keras에서는 map() 을 이용해 전처리 파이프라인을 추가할 수 있습니다.
🔥 Keras 전처리 코드
def preprocess(image, label):
    image = tf.image.resize(image, (224, 224))
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

train_ds = train_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
3. Deep Learning Modeling (모델 구성)
아래는 간단한 CNN 기반 분류 모델(Keras Sequential 사용) 예시입니다.
🔥 Keras 모델 구성 코드
from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)),
    layers.MaxPooling2D(4),

    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D(4),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
4. Compiler Setting (컴파일러 설정)
Keras의 compile() API를 사용하여 Optimizer, Loss, Metrics 등을 설정합니다.
🔥 Keras 컴파일 코드
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
5. Training Setting (학습 설정)
학습 반복 횟수(Epoch)를 설정합니다.
epochs = 5
6. Start Training (학습 시작)
아래는 학습을 진행하는 전체 코드입니다.
🚀 Keras 전체 학습 코드 (FULL VERSION)
import tensorflow as tf
from tensorflow.keras import layers, models

# -----------------------------
# 1) Dataset Loader
# -----------------------------
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset/train",
    image_size=(224, 224),
    batch_size=32
)

# -----------------------------
# 2) Data Preprocessing
# -----------------------------
def preprocess(image, label):
    image = tf.image.resize(image, (224, 224))
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

train_ds = train_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

# -----------------------------
# 3) Deep Learning Model
# -----------------------------
model = models.Sequential([
    layers.Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)),
    layers.MaxPooling2D(4),

    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D(4),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# -----------------------------
# 4) Compiler Setting
# -----------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# -----------------------------
# 5) Training Setting
# -----------------------------
epochs = 5

# -----------------------------
# 6) Start Training
# -----------------------------
print("Training Started...")
history = model.fit(train_ds, epochs=epochs)
print("Training Completed!")
