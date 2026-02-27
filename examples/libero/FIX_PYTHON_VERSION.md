# 解决 Python 版本兼容性问题

## 问题
`torch==1.11.0+cu113` 不支持 Python 3.11，只支持 Python 3.7-3.10。

## 解决方案

### 方案 1：使用 Python 3.8 环境（推荐）

根据 README，应该使用 Python 3.8：

```bash
# 创建 Python 3.8 虚拟环境
uv venv --python 3.8 examples/libero/.venv

# 激活环境
source examples/libero/.venv/bin/activate

# 安装依赖
uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu113 \
    --index-strategy=unsafe-best-match

# 安装其他包
uv pip install -e packages/openpi-client
uv pip install -e third_party/libero
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
```

### 方案 2：升级 PyTorch 版本（如果必须使用 Python 3.11）

如果必须使用 Python 3.11，需要升级 PyTorch 版本：

1. 修改 `examples/libero/requirements.in`：
   ```txt
   torch>=2.0.0
   torchvision>=0.15.0
   torchaudio>=2.0.0
   ```

2. 重新编译 requirements.txt：
   ```bash
   uv pip compile examples/libero/requirements.in \
       -o examples/libero/requirements.txt \
       --python-version 3.11 \
       --extra-index-url https://download.pytorch.org/whl/cu113 \
       --index-strategy=unsafe-best-match
   ```

**注意**：升级 PyTorch 可能导致与其他依赖（如 robosuite）不兼容。

### 方案 3：使用 conda 环境

```bash
# 创建 Python 3.8 conda 环境
conda create -n openpi_libero python=3.8
conda activate openpi_libero

# 安装 PyTorch（CUDA 11.3）
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 \
    cudatoolkit=11.3 -c pytorch

# 然后安装其他依赖
uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu113 \
    --index-strategy=unsafe-best-match
```

## 推荐方案

**强烈推荐使用方案 1（Python 3.8）**，因为：
1. 与 README 文档一致
2. 与 requirements.txt 兼容
3. 避免版本冲突
4. 经过测试的配置
