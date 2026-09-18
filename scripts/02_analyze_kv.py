import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
os.makedirs("plots", exist_ok=True)
data = torch.load("outputs/kv_sample.pt", map_location="cpu")
rows = []
for layer, kv in data.items():
    for name in ["k", "v"]:
        x = kv[name].float()

        # [B, Hkv, T, D]
        for h in range(x.shape[1]):
            y = x[:, h].reshape(-1)
            rows.append({
                "layer": layer,
                "type": name.upper(),
                "head": h,
                "min": y.min().item(),
                "max": y.max().item(),
                "amax": y.abs().max().item(),
                "mean": y.mean().item(),
                "std": y.std().item(),
                "p99_abs": y.abs().quantile(0.99).item(),
                "p999_abs": y.abs().quantile(0.999).item(),
            })
        y = x.reshape(-1).numpy()
        plt.figure(figsize=(7, 4))
        plt.hist(y, bins=200)
        plt.title(f"Layer {layer} {name.upper()} distribution")
        plt.xlabel("value")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(f"plots/layer_{layer}_{name}.png", dpi=150)
        plt.close()
pd.DataFrame(rows).to_csv("outputs/kv_stats.csv", index=False)
print(pd.DataFrame(rows).to_string(index=False))
