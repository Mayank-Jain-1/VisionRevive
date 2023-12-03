import torch
import config

def save_checkpoint(model, optimizer, filename):
    print(" --------- Saving Model --------- ")
    checkpoint = {
        "model_dict": model.state_dict(),
        "optimizer_dict": optimizer.state_dict()
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_name, model, optimizer, lr):
    print(" --------- Loading Checkpoint ---------")
    checkpoint = torch.load(checkpoint_name, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["model_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_dict"])

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr