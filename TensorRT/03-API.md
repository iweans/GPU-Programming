# TensorRT API

## 核心概念

### 核心类

`Tensor API`主要由五种类对象组成，分别是：

* Logger（`tensorrt.Logger`）

  其他大多数的类会使用其进行`错误报告`、`警告`和`提示信息`。

* Network（`tensorrt.INetworkDefinition`）

  用于表示一个`计算图（computational graph）`。

* Context（`tensorrt.IExecutionContext`）

  用于操作`计算图`来执行`推理`，类似于`TensorFlow`中的`会话（Session）对象`。

* Engine（`tensorrt.ICudaEngine`）

  该类是`TensorRT`最主要的组成元素（我们最后的目标就是生成一个`推理引擎`）。

  我们可以通过`Engine`来生成`Context`。

* Builder（`tensorrt.Builder`）

  用于构建空的`Network`和`Engine`。

* Parser

  用于将经过训练框架训练过的模型填充进`Network`。



### 工作流

通常，`TensorRT`的`工作流（workflow）`包含三个步骤：

1. 使用`tensorrt.Builer`来生成（generate）一个空的`tensorrt.INetworkDefinition`
2. 填充（populate）`tensorrt.INetworkDefinition`
3. 使用`tensorrt.Builder`来构建（build）一个`tensorrt.ICudaEngine`，这个过程需要用到经过填充的`tensorrt.INetworkDefinition`
4. 从`tensorrt.ICudaEngine`创建一个`tensorrt.IExecutionContext`，用于执行`优化的推理（optimized inference）`



下面是根据该工作流实现的脚手架代码：

```python
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
# --------------------------------------------------

MAX_WORKSPACE_SIZE = ...
# ----------------------------------------
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
# --------------------------------------------------

def populate_network(network, filler): pass

def prepare_buffers(engine): pass

def do_inference(context, img, inputs, outputs, bindings, stream): pass

# --------------------------------------------------
with trt.Builder(TRT_LOGGER) as builder:
    builder.max_workspace_size = MAX_WORKSPACE_SIZE
    with builder.create_network() as network:
        populate_network(network, filler)
        with builder.build_cuda_engine(network) as engine:
            inputs, outputs, bindings, stream = prepare_buffers(engine)
        	with engine.create_execution_context() as context:
                pred = do_inference(context, img, inputs, outputs, bindings, stream)
```



