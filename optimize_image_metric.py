import argparse
import csv
import math
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from torchvision.utils import save_image, make_grid

import matplotlib.pyplot as plt
import lpips


# ------------------------------------------------------------
# Image utilities
# ------------------------------------------------------------

def load_image(path, resize=None, device="cuda"):
    img = Image.open(path).convert("RGB")

    if resize is not None:
        img = img.resize((resize, resize), Image.BICUBIC)

    tf = transforms.ToTensor()  # [0, 1], shape [3, H, W]
    x = tf(img).unsqueeze(0).to(device)

    return x


def save_tensor_image(x, path):
    x = x.detach().clamp(0, 1).cpu()
    save_image(x, path)


def psnr_from_mse(mse):
    if mse <= 1e-12:
        return float("inf")
    return -10.0 * math.log10(mse)


def make_comparison_grid(start, target, final, path):
    """
    Saves a visual grid:
    [start | target | final | amplified absolute error]
    """
    with torch.no_grad():
        err = torch.abs(final - target)
        err_vis = (4.0 * err).clamp(0, 1)

        grid = make_grid(
            torch.cat([
                start.detach().cpu(),
                target.detach().cpu(),
                final.detach().cpu(),
                err_vis.detach().cpu(),
            ], dim=0),
            nrow=4,
            padding=4,
        )

        save_image(grid, path)


# ------------------------------------------------------------
# Metrics / losses
# ------------------------------------------------------------

class PixelDistance(nn.Module):
    def __init__(self, kind):
        super().__init__()
        assert kind in ["mse", "l1"]
        self.kind = kind

    def forward(self, x0, x1):
        if self.kind == "mse":
            return ((x0 - x1) ** 2).mean(dim=[1, 2, 3])
        else:
            return torch.abs(x0 - x1).mean(dim=[1, 2, 3])


class LPIPSVGGDistance(nn.Module):
    """
    Official LPIPS-VGG.

    This wrapper receives tensors in [0, 1] and internally converts to [-1, 1],
    as expected by the official LPIPS implementation.
    """

    def __init__(self):
        super().__init__()
        self.metric = lpips.LPIPS(net="vgg")

        for p in self.metric.parameters():
            p.requires_grad = False

        self.metric.eval()

    def forward(self, x0, x1):
        x0 = x0 * 2.0 - 1.0
        x1 = x1 * 2.0 - 1.0
        return self.metric(x0, x1).view(-1)


class VGGFeatureDistance(nn.Module):
    """
    Custom VGG16 feature distance.

    Input:
    - tensors in [0, 1]
    - shape [N, 3, H, W]

    Candidates:
    - vgg_r12 = relu1_2
    - vgg_r43 = relu4_3

    This is intentionally simple:
    - ImageNet normalization;
    - frozen VGG16;
    - optional LPIPS-style feature normalization;
    - mean squared distance in feature space.
    """

    LAYERS = {
        "relu1_2": 3,
        "relu2_2": 8,
        "relu3_3": 15,
        "relu4_3": 22,
        "relu5_3": 29,
    }

    CANDIDATES = {
        "vgg_r12": ["relu1_2"],
        "vgg_r43": ["relu4_3"],
    }

    def __init__(self, candidate_name, normalize_features=True):
        super().__init__()

        if candidate_name not in self.CANDIDATES:
            raise ValueError(f"Unknown VGG candidate: {candidate_name}")

        self.candidate_name = candidate_name
        self.layer_names = self.CANDIDATES[candidate_name]
        self.layer_ids = [self.LAYERS[name] for name in self.layer_names]
        self.max_layer = max(self.layer_ids)
        self.normalize_features = normalize_features

        try:
            weights = models.VGG16_Weights.IMAGENET1K_V1
            vgg = models.vgg16(weights=weights).features
        except AttributeError:
            vgg = models.vgg16(pretrained=True).features

        self.vgg = vgg[: self.max_layer + 1].eval()

        for p in self.vgg.parameters():
            p.requires_grad = False

        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )

        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def _imagenet_norm(self, x):
        return (x - self.mean) / self.std

    @staticmethod
    def _normalize_feat(f, eps=1e-10):
        norm = torch.sqrt(torch.sum(f ** 2, dim=1, keepdim=True))
        return f / (norm + eps)

    def _extract(self, x):
        feats = {}
        x = self._imagenet_norm(x)

        for i, layer in enumerate(self.vgg):
            x = layer(x)

            if i in self.layer_ids:
                feats[i] = x

        return feats

    def forward(self, x0, x1):
        f0 = self._extract(x0)
        f1 = self._extract(x1)

        total = 0.0

        for layer_id in self.layer_ids:
            a = f0[layer_id]
            b = f1[layer_id]

            if self.normalize_features:
                a = self._normalize_feat(a)
                b = self._normalize_feat(b)

            d = (a - b) ** 2
            d = d.mean(dim=[1, 2, 3])

            total = total + d

        return total / len(self.layer_ids)


def make_metric(name, device):
    if name == "mse":
        metric = PixelDistance("mse")
    elif name == "l1":
        metric = PixelDistance("l1")
    elif name == "lpips_vgg":
        metric = LPIPSVGGDistance()
    elif name in ["vgg_r12", "vgg_r43"]:
        metric = VGGFeatureDistance(name, normalize_features=True)
    else:
        raise ValueError(f"Unknown metric: {name}")

    metric = metric.to(device)
    metric.eval()

    for p in metric.parameters():
        p.requires_grad = False

    return metric


# ------------------------------------------------------------
# Optimization
# ------------------------------------------------------------

def make_initial_image(start_mode, target, input_img=None, noise_std=0.1):
    if start_mode == "input":
        if input_img is None:
            raise ValueError("start_mode='input' requires --input")
        x0 = input_img.clone()

    elif start_mode == "noisy_target":
        x0 = target + noise_std * torch.randn_like(target)
        x0 = x0.clamp(0, 1)

    elif start_mode == "random":
        x0 = torch.rand_like(target)

    else:
        raise ValueError(f"Unknown start mode: {start_mode}")

    return x0.detach()


@torch.no_grad()
def compute_eval_metrics(x, target, lpips_eval):
    mse = F.mse_loss(x, target).item()
    l1 = F.l1_loss(x, target).item()
    psnr = psnr_from_mse(mse)
    lpips_vgg = lpips_eval(x, target).mean().item()

    return {
        "mse": mse,
        "l1": l1,
        "psnr": psnr,
        "lpips_vgg": lpips_vgg,
    }


def optimize_one_run(
    metric_name,
    start_mode,
    metric,
    target,
    input_img,
    lpips_eval,
    outdir,
    steps=1000,
    lr=0.03,
    noise_std=0.1,
    save_every=100,
):
    outdir.mkdir(parents=True, exist_ok=True)

    x_start = make_initial_image(
        start_mode=start_mode,
        target=target,
        input_img=input_img,
        noise_std=noise_std,
    )

    x = x_start.clone().detach()
    x.requires_grad_(True)

    optimizer = torch.optim.Adam([x], lr=lr)

    history = []

    save_tensor_image(x_start, outdir / "start.png")
    save_tensor_image(target, outdir / "target.png")

    for step in range(steps + 1):
        optimizer.zero_grad(set_to_none=True)

        loss = metric(x, target).mean()
        loss.backward()

        grad_norm = x.grad.detach().norm().item()

        optimizer.step()

        # Simple valid-range constraint.
        # This keeps x as an actual image tensor in [0, 1].
        with torch.no_grad():
            x.clamp_(0, 1)

        loss_value = loss.item()

        if step % save_every == 0 or step == steps:
            eval_metrics = compute_eval_metrics(x.detach(), target, lpips_eval)

            row = {
                "step": step,
                "objective": loss_value,
                "grad_norm": grad_norm,
                **eval_metrics,
            }

            history.append(row)

            save_tensor_image(x, outdir / f"step_{step:05d}.png")

            print(
                f"[{metric_name:9s} | {start_mode:12s}] "
                f"step {step:5d} | "
                f"obj={loss_value:.6f} | "
                f"mse={eval_metrics['mse']:.6f} | "
                f"psnr={eval_metrics['psnr']:.2f} | "
                f"lpips_vgg={eval_metrics['lpips_vgg']:.6f} | "
                f"grad={grad_norm:.3e}"
            )

    final = x.detach().clamp(0, 1)

    save_tensor_image(final, outdir / "final.png")
    make_comparison_grid(x_start, target, final, outdir / "comparison_grid.png")

    # Save history CSV
    history_path = outdir / "history.csv"
    with open(history_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    # Plot objective curve
    steps_list = [h["step"] for h in history]
    obj_list = [h["objective"] for h in history]

    plt.figure()
    plt.plot(steps_list, obj_list)
    plt.xlabel("Step")
    plt.ylabel("Objective loss")
    plt.title(f"{metric_name} optimization from {start_mode}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outdir / "loss_curve.png", dpi=150)
    plt.close()

    final_metrics = compute_eval_metrics(final, target, lpips_eval)

    return {
        "metric": metric_name,
        "start": start_mode,
        "final_objective": history[-1]["objective"],
        "final_grad_norm": history[-1]["grad_norm"],
        **final_metrics,
        "outdir": str(outdir),
    }


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--input", type=str, default=None)

    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["vgg_r12", "vgg_r43", "lpips_vgg", "mse", "l1"],
        choices=["vgg_r12", "vgg_r43", "lpips_vgg", "mse", "l1"],
    )

    parser.add_argument(
        "--starts",
        nargs="+",
        default=["input", "noisy_target", "random"],
        choices=["input", "noisy_target", "random"],
    )

    parser.add_argument("--resize", type=int, default=256)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--noise_std", type=float, default=0.1)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--outdir", type=str, default="direct_optimization_results")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available. Falling back to CPU.")
        args.device = "cpu"

    device = torch.device(args.device)

    target = load_image(args.target, resize=args.resize, device=device)

    input_img = None
    if args.input is not None:
        input_img = load_image(args.input, resize=args.resize, device=device)

    if "input" in args.starts and input_img is None:
        print("Warning: --starts contains 'input' but --input was not provided. Skipping input start.")
        args.starts = [s for s in args.starts if s != "input"]

    root_outdir = Path(args.outdir)
    root_outdir.mkdir(parents=True, exist_ok=True)

    # LPIPS-VGG used only for reporting, independent of the optimization objective.
    lpips_eval = LPIPSVGGDistance().to(device).eval()

    summary_rows = []

    for metric_name in args.metrics:
        print("")
        print("=" * 80)
        print(f"Metric objective: {metric_name}")
        print("=" * 80)

        metric = make_metric(metric_name, device)

        for start_mode in args.starts:
            run_outdir = root_outdir / f"{metric_name}__start_{start_mode}"

            row = optimize_one_run(
                metric_name=metric_name,
                start_mode=start_mode,
                metric=metric,
                target=target,
                input_img=input_img,
                lpips_eval=lpips_eval,
                outdir=run_outdir,
                steps=args.steps,
                lr=args.lr,
                noise_std=args.noise_std,
                save_every=args.save_every,
            )

            summary_rows.append(row)

        del metric

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary_path = root_outdir / "summary.csv"

    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print("")
    print("Saved summary to:", summary_path)
    print("Done.")


if __name__ == "__main__":
    main()