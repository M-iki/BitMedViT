<div align="center">
  <h1 style="font-family: Arial;">BitMedViT</h1>
  <h3>BitMedViT: Ternary Quantized Vision Transformer for Medical AI Assistants on the Edge</h3>
</div>
<p align="center"> <img src="images/bitmedvit_doctor_logo.png" alt="BitMedViT logo" width="40%"> </p>

## Abstract
Vision Transformers (ViTs) have demonstrated strong capabilities in interpreting complex medical imaging data. However, their significant computational and memory demands pose challenges for deployment in real-time, resource-constrained mobile and wearable devices used in clinical environments. We introduce, BiTMedViT, a new class of Edge ViTs serving as medical AI assistants that perform structured analysis of medical images directly on the edge. BiTMedViT utilizes ternary- quantized linear layers tailored for medical imaging and com- bines a training procedure with multi-query attention, preserving stability under ternary weights with low-precision activations. Furthermore, BiTMedViT employs task-aware distillation from a high-capacity teacher to recover accuracy lost due to extreme quantization. Lastly, we also present a pipeline that maps the ternarized ViTs to a custom CUDA kernel for efficient memory bandwidth utilization and latency reduction on the Jetson Orin Nano. Finally, BiTMedViT achieves 86% diagnostic accuracy (89% SOTA) on MedMNIST across 12 datasets, while reducing model size by 43x, memory traffic by 39x, and enabling 16.8 ms inference at an energy efficiency up to 41x that of SOTA models at 183.62 GOPs/J on the Orin Nano. Our results demonstrate a practical and scientifically grounded route for extreme-precision medical imaging ViTs deployable on the edge, narrowing the gap between algorithmic advances and deployable clinical tools.

## TL;DR

* Edge-ready Vision Transformer tailored for medical image classification
* Ternary weights with low-precision activations for efficient inference
* Stable training via multi-query attention and task-aware knowledge distillation
* Custom CUDA + TensorRT deployment optimized for NVIDIA Jetson Orin Nano

## Getting Started

### Training

### Evaluation

### Custom Kernel and TensorRT Deployment

## References
* [MedViTv2](https://github.com/Omid-Nejati/MedViTV2)
* [BitNet](https://github.com/microsoft/BitNet)
* [Vit-1.58b](https://github.com/DLYuanGod/ViT-1.58b)
* [BitNetTransformer](https://github.com/kyegomez/BitNet)

## Citation
```bibtex
@misc{walczak2025bitmedvit,
      title={Invited Paper: BitMedViT: Ternary-Quantized Vision Transformer for Medical AI Assistants on the Edge}, 
      author={Mikolaj Walczak and Uttej Kallakuri and Edward Humes and Xiaomin Lin and Tinoosh Mohsenin},
      year={2025},
      eprint={2510.13760},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2510.13760}, 
}
```
