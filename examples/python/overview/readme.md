📌 딥러닝 학습 전체 워크플로우
(PyTorch & TensorFlow 함수 + 샘플 코드 포함)
이 문서는 PyTorch 또는 TensorFlow 기반 딥러닝을 어떻게 진행하는지에 대한 정보를 담고 있습니다.

전체 과정은 아래 순서대로 진행됩니다.
Dataset Loader
Data Preprocessing
Deep Learning Modeling
Compiler Setting
Training Setting
Start Training

1. Dataset Loader (데이터셋 로더)
Dataset 로딩은 TensorFlow/PyTorch의 기본 API 또는 일반 Python 라이브러리를 통해 가능합니다.
다만, TF/PyTorch Dataset Loader를 사용할 경우 프레임워크에서 요구하는 디렉토리 구조를 따라야 합니다.
🔥 PyTorch 샘플 코드
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_dataset = datasets.ImageFolder("dataset/train", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
🔥 TensorFlow 샘플 코드
import tensorflow as tf

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset/train",
    image_size=(224, 224),
    batch_size=32
)

train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

2. Data Preprocessing (데이터 전처리)
입력 데이터는 모델 성능을 높이기 위해 정규화 및 크기 변환 등의 전처리가 필요합니다.
정규화(normalization)의 목적은 데이터 값의 편차를 줄여 학습 안정성을 확보하는 것입니다.
🔥 PyTorch 샘플 코드
from torchvision import transforms

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
🔥 TensorFlow 샘플 코드
import tensorflow as tf

def preprocess(image, label):
    image = tf.image.resize(image, (224, 224))
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

train_ds = train_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)

3. Deep Learning Modeling (모델 구성)
Classification / Detection / Pose 등 다양한 문제 유형에 맞는 모델을 구성합니다.
🔥 PyTorch 샘플 코드
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.fc1 = nn.Linear(16 * 56 * 56, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = nn.MaxPool2d(4)(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

model = SimpleCNN()
🔥 TensorFlow 샘플 코드
from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)),
    layers.MaxPooling2D(4),
    layers.Flatten(),
    layers.Dense(10)
])

4. Compiler Setting (컴파일러 설정)
Optimizer, Loss 함수 등 학습 과정에서 필요한 컴파일 옵션을 설정합니다.
🔥 PyTorch 샘플 코드
import torch.optim as optim
import torch.nn as nn

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
🔥 TensorFlow 샘플 코드
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

5. Training Setting (학습 설정)
Epoch: 전체 dataset을 몇 번 반복할지
Batch size: 한 번에 몇 개의 데이터를 모델에 넣을지
설정 값들은 학습 속도 및 메모리 사용량에 직접적인 영향을 줍니다.
🔥 PyTorch 샘플 코드
num_epochs = 5

for epoch in range(num_epochs):
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {loss.item():.4f}")
🔥 TensorFlow 샘플 코드
history = model.fit(train_ds, epochs=5)

6. Start Training (학습 시작)
모든 설정이 완료되면 학습을 시작합니다.
🔥 PyTorch 샘플 코드
print("Training Started...")
for epoch in range(5):
    ...
print("Training Completed!")
🔥 TensorFlow 샘플 코드
model.fit(train_ds, epochs=5)
print("Training Completed!")



