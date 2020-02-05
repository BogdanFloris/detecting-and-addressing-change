import torch
from models.wos_train import train_wos_stream

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Running on {}".format(device))
    train_wos_stream(epochs=5, transform=False, device=device)

