# BitNet Inference Kernel

This repository provides a highly efficient GEMV kernel implementation for the BitNet model, optimized for W2A8 inference — 2-bit weights and 8-bit activations. It is tailored for use with the [BitNet-b1.58-2B-4T](https://arxiv.org/abs/2504.12285) model.

## Features

- Support for W2A8 (2-bit weight × 8-bit activation) GEMV computation  
- Custom CUDA kernels with low-latency execution  
- Optimizations for memory access, decoding, and compute throughput  

## Usage

Installation and kernel performance tests:

```bash
# (Recommended) Create a new conda environment
conda create --name bitnet-gpu "python<3.13"
conda activate bitnet-gpu

# Install dependencies
pip install -r requirements.txt

# Build the kernel
cd bitnet_kernels
bash compile.sh
cd ..

# Run performance tests
python test.py
```

## Optimizations

## Performance

### Kernel Benchmarks

