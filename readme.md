

<p align="left">
  <img src="https://raw.githubusercontent.com/emtake-ai/eagleboard/main/folder.png" width="30%">
</p>

📁 Project Directory Description

이 리포지토리는 Synabro 기반 AI/NPU 시스템 개발을 위해 구성된 구조로,
하드웨어·모델 변환·SDK·예제 코드·튜토리얼 등을 명확하게 분리하여 관리한다.
아래는 각 디렉토리의 역할을 설명한 내용이다.

📚 docs/

프로젝트 전반의 문서를 저장하는 디렉토리입니다.

● docs/hardware/

하드웨어 관련 문서를 저장하는 디렉토리입니다.
보드 스펙, 인터페이스 설명, 핀맵 등 HW 정보를 포함합니다.

● docs/model_conversion/

Synabro를 사용하여 AI 모델(Keras/PyTorch 등)을 NPU 실행 가능한 .lne 파일로 변환하는 방법을 제공하는 문서 디렉토리입니다.
모델 변환 과정, quantization, onnx 변환 절차 등이 포함됩니다.

● docs/npu/

NPU 구조와 동작 방식에 대한 문서를 담는 디렉토리입니다.
NPU의 내부 구조, 연산 방식, 스케줄링, 메모리 구조 등을 설명합니다.

● docs/software/

소프트웨어(SDK, runtime, compiler 등)를 어떻게 사용하는지 설명하는 문서를 제공하는 디렉토리입니다.
API 사용법, 환경 설정, 실행 절차 등이 포함됩니다.

● docs/tutorials/

프로젝트를 처음 시작하는 사용자를 위한 Getting Started 튜토리얼 문서가 있는 디렉토리입니다.
초보자가 따라 할 수 있는 실습 가이드가 포함됩니다.

🧪 examples/

예제 코드 모음 디렉토리입니다.
C 기반 예제, Python 예제, Edge 디바이스용 예제가 포함됩니다.

● examples/C/

C 언어 기반 예제 프로그램이 포함된 디렉토리입니다.
임베디드 환경에서 NPU를 활용하는 기본 예제를 포함합니다.

● examples/python/

Python 기반 예제로, SDK를 Python으로 사용하는 방법을 보여줍니다.

● examples/edge/

Edge 환경에서 모델을 배포(deployment)하는 방법을 설명하는 디렉토리입니다.
센서 → 전처리 → NPU inference → 결과 전송 등의 플로우를 예제로 제공합니다.

● examples/models/

AI 모델 코드가 포함된 디렉토리로 Keras와 PyTorch 모델을 각각 분리하여 관리합니다.

examples/models/Keras/
Keras 기반 모델의 Python 코드 및 예제가 포함됨.

examples/models/Pytorch/
PyTorch 기반 모델의 Python 코드 및 예제가 포함됨.

🔧 sdk/

Synabro SDK 관련 코드와 실행 런타임이 존재하는 디렉토리입니다.
모델을 실행하고 관리하는 핵심 기능들이 이곳에 포함됩니다.

● sdk/runtime/

NPU 모델을 실제로 실행하는 런타임 엔진이 포함된 디렉토리입니다.

● sdk/synabro/

Synabro 툴체인, Python/C Wrappers, 모델 변환 및 실행을 위한 유틸리티 등이 포함된 디렉토리입니다.
