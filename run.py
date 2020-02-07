import torch
from streams.drift_experiments import drift_detection_different_embeddings
from models.wos_train import train_wos_batch

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Running on {}".format(device))
    try:
        # drift_detection_different_embeddings(batch_size=32, device=device)
        train_wos_batch(epochs=5, device=device)
    except Exception as e:
        print(e.with_traceback(e.__traceback__))
