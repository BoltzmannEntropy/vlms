# Dockerfile customized for deployment on HuggingFace Spaces platform

# -- The Dockerfile has been tailored specifically for use on HuggingFace.
# -- It implies that certain modifications or optimizations have been made with HuggingFace's environment in mind.
# -- It uses "HuggingFace Spaces" to be more specific about the target platform.

# FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-devel
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel 
# FOR HF

USER root

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    python3-numpy \
    gcc \
    build-essential \
    gfortran \
    wget \
    curl \
    pkg-config \
    software-properties-common \
    zip \
    && apt-get clean && rm -rf /tmp/* /var/tmp/*

RUN apt-get update && DEBIAN_FRONTEND=noninteractive \
    apt-get install -y python3.10 python3-pip

RUN apt-get install -y libopenblas-base libopenmpi-dev

ENV TZ=Asia/Dubai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone



RUN useradd -m -u 1000 user
    
RUN apt-get update && apt-get install -y sudo && \
echo 'user ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

USER user
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

RUN mkdir $HOME/app
RUN mkdir $HOME/app/test_images

# WORKDIR $HOME/app

RUN chown -R user:user $HOME/app   

USER user
WORKDIR $HOME/app

RUN python -m  pip install  qwen-vl-utils 
RUN python -m pip install --pre -U -f https://mlc.ai/wheels mlc-llm-nightly-cu122 mlc-ai-nightly-cu122

RUN  python3 -m pip install chromadb db-sqlite3 auto-gptq exllama sqlalchemy  
WORKDIR $HOME/app
RUN git clone https://github.com/casper-hansen/AutoAWQ  
WORKDIR $HOME/app/AutoAWQ/ 
RUN python3 -m pip  install -e .
WORKDIR $HOME/app
# ENV FLASH_ATTENTION_FORCE_BUILD=TRUE
RUN python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN python -m pip install  accelerate diffusers datasets timm flash-attn==2.6.1 gradio

RUN  python3 -m pip install --no-deps optimum  
RUN  python3 -m pip install --no-deps autoawq>=0.1.8 

#This seems to be a must : Intel Extension for PyTorch 2.4 needs to work with PyTorch 2.4.*, but PyTorch 2.2.2 is
RUN python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN  python3 -m pip install -U accelerate 
RUN  python3 -m pip install -U git+https://github.com/huggingface/transformers

WORKDIR $HOME/app
COPY --chown=user:user app.py .
COPY --chown=user:user test_images /home/user/app/test_images
# /home/user/app/
# chown -R user:user /home/user/.cache/

ENV PYTHONUNBUFFERED=1 	GRADIO_ALLOW_FLAGGING=never 	GRADIO_NUM_PORTS=1 	GRADIO_SERVER_NAME=0.0.0.0     GRADIO_SERVER_PORT=7860 	SYSTEM=spaces
RUN python3 -m pip  install pennylane sympy pennylane-qiskit duckdb
WORKDIR $HOME/app

EXPOSE 8097 7842 8501 8000 6666 7860

CMD ["python", "app.py"]


# ERROR! Intel® Extension for PyTorch* needs to work with PyTorch 2.4.*, but PyTorch 2.2.2 is found. Please switch to the matching version and run again.
# ERROR! Intel® Extension for PyTorch* needs to work with PyTorch 2.4.*, but PyTorch 2.2.2 is found. Please switch to the matching version and run again.
# `Qwen2VLRotaryEmbedding` can now be fully parameterized by passing the model config through the `config` argument. All other arguments will be removed in v4.46
# /home/user/.local/lib/python3.10/site-packages/transformers/modeling_utils.py:4749: FutureWarning: `_is_quantized_training_enabled` is going to be deprecated in transformers 4.39.0. Please use `model.hf_quantizer.is_trainable` instead
#   warnings.warn(
# /home/user/.local/lib/python3.10/site-packages/accelerate/utils/imports.py:336: UserWarning: Intel Extension for PyTorch 2.4 needs to work with PyTorch 2.4.*, but PyTorch 2.2.2 is found. Please switch to the matching version and run again.
#   warnings.warn(
# Error loading model Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int4: Found modules on cpu/disk. 
# Using Exllama or Exllamav2 backend requires all the modules to be on GPU.You can deactivate exllama backend by setting `disable_exllama=True` in the quantization config object
# Error loading model Qwen/Qwen2-VL-7B-Instruct: (ReadTimeoutError("HTTPSConnectionPool(host='hf.co', port=443): 
# Read timed out. (read timeout=10)"), '(Request ID: b8269a88-9b6b-43e0-942d-1049f173dc00)')

# Error loading model Qwen/Qwen2-VL-7B-Instruct: CUDA out of memory. 
# Tried to allocate 130.00 MiB. GPU 0 has a total capacity of 14.58 GiB of which 77.62 MiB is free.

# instruct: FlashAttention only supports Ampere GPUs or newer.