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
        	with engine.create_execution_context() as context:
                inputs, outputs, bindings, stream = prepare_buffers(engine)
                pred = do_inference(context, img, inputs, outputs, bindings, stream)
```



## 填充网络

### 网络定义API

我们以一个“两层`卷积`+两层`全链接`”组成的简单卷积神经网络为例展开本节的讨论：

```python
def populate_network(network, weights):
    input = network.add_input(shape=(1, 28, 28), dtype=trt.float32, name='input')
    # ---------------------------------------- Conv01
    conv01 = network.add_convolution(input=input, kernel_shape=(5, 5), num_output_maps=20,
                                     kernel=np.array(weights['conv1.weight'], dtype='float32'),
                                     bias=np.array(weights['conv1.bias'], dtype='float32'))
    conv01.stride = (1, 1)
    pool01 = network.add_pooling(input=conv01.get_output(0), window_size=(2, 2), type=trt.PoolingType.MAX)
    pool01.stride = (2, 2)
    # ---------------------------------------- Conv02
    conv02 = network.add_convolution(input=pool01.get_output(0), kernel_shape=(5, 5), num_output_maps=50,
                                     kernel=np.array(weights['conv2.weight'], dtype='float32'),
                                     bias=np.array(weights['conv2.bias'], dtype='float32'))
    conv02.stride = (1, 1)
    pool02 = network.add_pooling(input=conv02.get_output(0), window_size=(2, 2), type=trt.PoolingType.MAX)
    pool02.stride = (2, 2)
    # ---------------------------------------- FC01
    fc01 = network.add_fully_connected(input=pool02.get_output(0), num_outputs=500,
                                       kernel=np.array(weights['fc1.weight'], dtype='float32'),
                                       bias=np.array(weights['fc1.bias'], dtype='float32'))
    activation01 = network.add_activation(input=fc01.get_output(0), type=trt.ActivationType.RELU)
    # ---------------------------------------- FC02
    fc02 = network.add_fully_connected(input=activation01.get_output(0), num_outputs=10,
                                       kernel=np.array(weights['fc2.weight'], dtype='float32'),
                                       bias=np.array(weights['fc2.bias'], dtype='float32'))
    # ---------------------------------------- Output
    fc02.get_output(0).name = 'logit'
    network.mark_output(tensor=fc02.get_output(0))
```



### 从UFF模型进行填充

```python
def populate_network(network, uff_path, input_name, input_shape, output_name):
    with trt.UffParser() as parser:
        parser.register_input(input_name, input_shape)
        parser.register_output(output_name)
        parser.parse(uff_path, network)
```



## 缓冲区操作

由于`TensorRT API`的定位是一个`低级别的API`，因此除了`网络运算的优化`以外没有什么是免费提供的。我们需要自己控制`内存`与`显存`之间的交互，这即给编码增加了难度，也释放了内存优化的空间。

这里，我们提供一个基础的示例来完成上诉工作，建议在应用到生产环境前对其进行完善和补充：

```python
class Buffer(object):

    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def alloc_buffers(engine, binding):
    max_batch_size = engine.max_batch_size
    binding_shape = engine.get_binding_shape(binding)
    binding_dtype = engine.get_binding_dtype(binding)
    host_buf = cuda.pagelocked_empty(
        trt.volume(binding_shape) * max_batch_size,
        trt.nptype(binding_dtype)
    )
    device_buf = cuda.mem_alloc(host_buf.nbytes)
    return Buffer(host_buf, device_buf)


def prepare_buffers(engine):
    inputs, outputs, bindings = list(), list(), list()
    stream = cuda.Stream()
    for binding in engine:
        binding_buf = alloc_buffers(engine, binding)
        bindings.append(binding_buf.device)
        if engine.binding_is_input(binding):
            inputs.append(binding_buf)
        else:
            outputs.append(binding_buf)

    return inputs, outputs, bindings, stream


def GiB(GiB_size):
    return GiB_size * 1 << 30
```

