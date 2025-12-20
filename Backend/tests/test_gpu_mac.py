import torch

def get_device():
    if torch.backends.mps.is_available():
        print("Folosesc GPU: Apple MPS (Metal)")
        return torch.device("mps")
    
    elif torch.cuda.is_available():
        print("Folosesc GPU: NVIDIA CUDA")
        return torch.device("cuda")
    
    else:
        print("Folosesc CPU")
        return torch.device("cpu")

device = get_device()

# model.to(device)