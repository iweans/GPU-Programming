# 安装

## 依赖

### 硬件依赖

由于`TensorRT`是基于`Nvidia GPU`开发的，因此首先需要的是一块达到版本算力的`Nvidia显卡`。

### 驱动依赖

* Nvidia GPU Driver
* CUDA
* cuDNN

### 软件依赖

* Python

* PyCUDA
* TensorFlow 1.x



## 获取TensorRT

下载相应的`Tar文件`后，对其进行解压，其中会包含如下一些子目录：

* bin：可执行文件
* include：C++头文件
* lib：C++库文件
* targets
* doc：产品文档
* python：Python包 — `Python API 封装`（仅支持`Linux`）
* graphsurge：Python包 — `TensorFlow Graph 操作工具`
* uff：Python包 — `TensorFlow Saved Model 转换工具`
* samples：一些示例（包括C++示例与Python示例）
* data：一些数据集相关的资源



## 配置TensorRT

```bash
export CUDA_HOME="/usr/local/cuda"
export CUDNN_INSTALL_DIR="$CUDA_HOME"
export TENSORRT_HOME="/usr/local/tensorrt"
export PATH="$TENSORRT_HOME/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$TENSORRT_HOME/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
```

激活新的环境变量：

```bash
$ source ~/.bashrc
```





