## Project Directory Description

This repository is structured for developing a Synabro-based AI/NPU system.  
Hardware documentation, model conversion, SDK, example code, and tutorials are clearly separated to ensure maintainability and scalability.  

Below is the description of each directory.  

### docs/

This directory contains all project documentation.  

###  docs/hardware/

Stores hardware-related documents such as board specifications, interface descriptions, pin maps, and other HW reference materials.  

###  docs/model_conversion/

Provides documentation on converting AI models (Keras / PyTorch) into NPU-executable .lne files using Synabro.  
It includes the full conversion pipeline, quantization, and ONNX export procedures.  

###  docs/npu/

Contains documentation describing the internal architecture and operation of the NPU, including computation flow, scheduling, and memory structure.  

###  docs/deployment/

Includes guides on how to deploy the model in EagelBoard  
It covers API usage, and execution procedures.  

###  docs/tutorials/

This directory provides Getting Started tutorials for first-time users.  
Step-by-step hands-on guides are included.  

### examples/

A collection of example codes.  

#### examples/C/  

Contains C-language based example programs demonstrating how to use the NPU in embedded environments.  

#### examples/python/

Python-based examples showing how to use the SDK with Python.  

#### examples/edge/

Provides examples of deploying models in edge environments.  
Typical flows such as sensor → preprocessing → NPU inference → result transmission are demonstrated.  

#### examples/models/

Contains AI model source codes, separated into Keras and PyTorch implementations.  

#### examples/models/Keras/
Keras-based model source code and examples.   

#### examples/models/Pytorch/
PyTorch-based model source code and examples.  

check below link with youtube

- [how to annotate in yolo using labelimg](https://www.youtube.com/watch?v=nV26hK3CxNM)
- [how to download the git repos using git clone](https://www.youtube.com/watch?v=SGCvhjD3mtM)
- [how to download the model from huggingface](https://www.youtube.com/watch?v=JCcyCxori0M)
- [how to build keras, and how to convert it with lne](https://www.youtube.com/watch?v=BDnK0pujDvg)
- [how to install synabro with docker](https://www.youtube.com/watch?v=fNOcj9eNf_M)
- [how to install yolov7 from github](https://www.youtube.com/watch?v=vVipUHJVF5o)

check on how to use V4L2 and ALSA devices

- [how to use ALSA device using Gstreamer](install_1.md)

- [how to use V4L2 device using Gstreamer](install_2.md)
