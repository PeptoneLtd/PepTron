# Use the BioNeMo Framework image as the base
FROM nvcr.io/nvidia/clara/bionemo-framework:2.3 AS openfold-bionemo-image
# Switch to root user to perform system-level operations
USER root
# Install build dependencies
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    ninja-build \
    git \
    git-lfs \
    wget \
    openssh-client && \
    rm -rf /var/lib/apt/lists/*
# Clone the openfold repository from GitHub
RUN git lfs install && \
    git clone -b nv_upstream_trt_cuequivariance --single-branch https://github.com/borisfom/openfold.git /openfold2
ENV PYTHONPATH="/openfold2"
# Set the working directory
WORKDIR /openfold2
# Install uv
RUN pip install uv
# Uninstall existing Triton and install Triton 3.3.0
# Install nvidia-ml-py (replacement for deprecated pynvml) for cuequivariance
RUN pip uninstall -y triton || true && \
    pip uninstall -y pynvml || true && \
    pip install triton==3.3.0 && \
    pip install nvidia-ml-py
# Install cuequivariance and its CUDA operations
RUN pip install cuequivariance_torch==0.6.1 && \
    pip install cuequivariance-ops-torch-cu12==0.6.1
# Install Python dependencies using uv
RUN uv pip install --upgrade pip && \
    uv pip install --no-cache-dir wheel setuptools && \
    uv pip install --no-cache-dir --no-build-isolation -e . && \
    uv pip install --no-cache-dir biopython==1.85 modelcif==1.5 ml_collections==1.1.0 bionemo-moco==0.0.2.2
RUN touch /openfold2/openfold/__init__.py
# Download stereo_chemical_props.txt
RUN wget https://git.scicore.unibas.ch/schwede/openstructure/-/raw/7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt && \
    mv stereo_chemical_props.txt /openfold2/openfold/resources/
# Verify the installation and Python path
RUN python -c "import sys; print(sys.path)" && \
    python -c "import openfold; print(openfold.__file__)"
# Set the default command to start a bash shell
CMD ["/bin/bash"]