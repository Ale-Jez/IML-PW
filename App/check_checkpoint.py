import torch
cp = torch.load('/checkpoints/train30/best_model.pt', map_location='cpu')
print('Top-level keys:', list(cp.keys())[:10])
if isinstance(cp, dict) and 'aamsm' in str(cp.keys()):
    print("\nHas aamsm")
if isinstance(cp, dict):
    for key in cp:
        if isinstance(cp[key], dict):
            print(f"{key}: dict with {len(cp[key])} items")
        elif isinstance(cp[key], torch.Tensor):
            print(f"{key}: tensor {cp[key].shape}")
        else:
            print(f"{key}: {type(cp[key])}")
