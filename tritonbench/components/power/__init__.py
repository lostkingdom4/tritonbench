import torch

# PowerManagerTask requires pynvml which is only available when NVIDIA GPU is present
if torch.cuda.is_available():
    from .power_manager import PowerManagerTask
else:
    PowerManagerTask = None
