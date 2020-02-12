import torch
from models.wos_train_nb import train_nb_wos_holdout

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Running on {}".format(device))
    try:
        train_nb_wos_holdout(epochs=5, transform=False, print_every=20, device=device)
    except Exception as e:
        print(e)
