import torch
from models.wos_train_lstm import train_lstm_wos_holdout

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Running on {}".format(device))
    try:
        train_lstm_wos_holdout(epochs=3, transform=False, print_every=1, device=device)
    except Exception as e:
        print(e)
