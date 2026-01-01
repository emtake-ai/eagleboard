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