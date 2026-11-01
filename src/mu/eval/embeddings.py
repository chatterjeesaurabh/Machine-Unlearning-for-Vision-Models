import torch

from mu.data import GPULoader


@torch.no_grad()
def extract(model, data, part, indices, batch_size=1000):
    """512-d penultimate (avgpool) embeddings, fp32, no augmentation. Returns (features [N,512] on GPU, labels [N])."""
    model.eval()
    feats, labels = [], []
    for x, y, _ in GPULoader(data, part, indices, batch_size):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            f = model.features(x)
        feats.append(f.float())
        labels.append(y)
    return torch.cat(feats), torch.cat(labels)
