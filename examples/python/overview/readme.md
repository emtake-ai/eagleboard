📌 딥러닝 학습 전체 워크플로우

이 프로젝트는 PyTorch 또는 TensorFlow 기반의 표준 딥러닝 학습 절차를 따릅니다.
전체 학습 과정은 다음과 같은 단계로 구성됩니다.
1. Dataset Loader
2. Data Preprocessing
3. Deep Learning Modeling
4. Compiler Setting
5. Training Setting
6. Start Training

각 단계에 대한 설명은 아래와 같습니다.

1. Dataset Loader (데이터셋 로더)
Dataset을 불러오는 방법은 크게 두 가지가 있습니다.
TensorFlow / PyTorch의 내장 Dataset 함수를 사용하는 방법
Python 표준 라이브러리(os, glob, PIL 등)를 이용해 직접 로드하는 방법
단, TensorFlow와 PyTorch의 Dataset Loader를 사용할 경우 프레임워크가 요구하는 디렉토리 구조를 따라야 합니다.
따라서 데이터셋을 올바른 구조로 재배치해야 정상적으로 로딩이 가능합니다.

2. Data Preprocessing (데이터 전처리)
모델 입력 전에 데이터를 반드시 전처리해야 합니다.
특히 정규화(Normalization) 는 입력값의 편차를 줄여 학습이 더 안정적으로 진행되도록 도와줍니다.
전처리 과정은 다음과 같은 작업을 포함할 수 있습니다.
이미지 크기 변환 (resize)
정규화
데이터 타입 변환
Augmentation(필요 시)

3. Deep Learning Modeling (모델 구성)
문제 유형에 따라 다양한 딥러닝 모델을 구성할 수 있습니다.
Classification(분류)
Detection(객체 탐지)
Pose Estimation(포즈 추정)
TensorFlow/Keras 또는 PyTorch 모듈을 활용해 모델 아키텍처를 정의합니다.

4. Compiler Setting (컴파일러 설정)
학습을 시작하기 전 다음과 같은 학습 관련 옵션을 설정해야 합니다.
Optimizer (예: Adam, SGD)
Loss 함수
Learning rate
기타 학습 하이퍼파라미터
이 설정들은 모델의 가중치를 어떻게 업데이트할지 결정하는 중요한 요소입니다.

5. Training Setting (학습 설정)
학습을 진행하기 위해 다음과 같은 값들을 지정해야 합니다.
Epoch
전체 데이터셋을 몇 번 반복할 것인지 설정
Batch size
한 번에 GPU/CPU에 투입되는 데이터 샘플 수
이 값들은 학습 속도, 메모리 사용량, 학습 안정성에 직접적인 영향을 줍니다.

6. Start Training (학습 시작)
모든 환경 설정이 완료되면 모델 학습을 시작합니다.
모델은 설정된 Epoch와 Batch size에 따라 데이터셋을 반복 학습하며 파라미터를 최적화합니다.