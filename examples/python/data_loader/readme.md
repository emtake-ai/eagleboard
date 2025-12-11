📌 Dataset Loader (PyTorch & TensorFlow 비교 버전)
딥러닝 학습에서 데이터셋을 불러오는 가장 기본 단계입니다.
PyTorch와 TensorFlow(Keras)에서는 서로 다른 API를 사용합니다.

🔥 1. PyTorch Dataset Loader
PyTorch는 torchvision.datasets.ImageFolder와 DataLoader를 사용하며,
디렉토리는 다음과 같은 구조를 가정합니다:
dataset/
 └── train/
      ├── class1/
      ├── class2/
      └── ...
✔️ PyTorch 코드 예시
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_dataset = datasets.ImageFolder("dataset/train", transform=transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,        # 옵션
    pin_memory=True       # 옵션 (GPU 사용 시 권장)
)

🔥 2. TensorFlow (Keras) Dataset Loader
TensorFlow는 image_dataset_from_directory() API를 사용하며,
디렉토리 구조는 PyTorch와 동일합니다:
dataset/
 └── train/
      ├── class1/
      ├── class2/
      └── ...
✔️ TensorFlow(Keras) 코드 예시
import tensorflow as tf

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset/train",
    image_size=(224, 224),   # 자동 resize
    batch_size=32,
    shuffle=True
)

# 성능 최적화
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

📌 핵심 차이 요약
기능	PyTorch	TensorFlow(Keras)
기본 Dataset API	ImageFolder	image_dataset_from_directory
반환 형태	Python Iterable (DataLoader)	tf.data.Dataset
전처리	transforms.Compose	map(preprocess_fn)
성능 설정	num_workers, pin_memory	AUTOTUNE, prefetch