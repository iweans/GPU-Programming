# TensorRT API

## 核心概念

### 核心类

* Logger
* Engine 与 Context
* Builder
* Network
* Parsers

### 工作流

通常，`TensorRT`的`工作流（workflow）`包含三个步骤：

1. 使用`tensorrt.Builer`来生成（generate）一个空的`tensorrt.INetworkDefinition`
2. 填充（populate）`tensorrt.INetworkDefinition`
3. 使用`tensorrt.Builder`来构建（build）一个`tensorrt.ICudaEngine`，这个过程需要用到经过填充的`tensorrt.INetworkDefinition`
4. 从`tensorrt.ICudaEngine`创建一个`tensorrt.IExecutionContext`，用于执行`优化的推理（optimized inference）`

