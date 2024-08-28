[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# Vitar Implementation
Implementation of the paper: "ViTAR: Vision Transformer with Any Resolution"

## Install

```
$ pip3 install -U vitar
```

## Example
````python
import torch
from vitar.main import Vitar

# Create a random input tensor
x = torch.randn(1, 3, 224, 224)

# Initialize the Vitar model with specified parameters
model = Vitar(
    512,
    8,
    depth=12,
    patch_size=16,
    image_size=224,
    channels=3,
    ffn_dim=2048,
    num_classes=1000,
)

# Pass the input tensor through the model
out = model(x)

# Print the output tensor
print(out)
```

## Citation
```bibtex
@misc{fan2024vitarvisiontransformerresolution,
    title={ViTAR: Vision Transformer with Any Resolution}, 
    author={Qihang Fan and Quanzeng You and Xiaotian Han and Yongfei Liu and Yunzhe Tao and Huaibo Huang and Ran He and Hongxia Yang},
    year={2024},
    eprint={2403.18361},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2403.18361}, 
}
```