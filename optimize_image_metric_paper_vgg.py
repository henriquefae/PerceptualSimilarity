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


try:
    import lpips
    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False


# ------------------------------------------------------------
# Image utilities
# ------------------------------------------------------------

def load_image(path, resize=None, device="cuda"):
    img = Image.open(path).convert("RGB")

    if resize is not None:
        img = img.resize((resize, resize), Image.BICUBIC)

    x = transforms.ToTensor()(img).unsqueeze(0).to(device)  # [1, 3, H, W], [0, 1]
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
    Saves:
    [input/start | target | final optimized image | 4x absolute error]
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
# LPIPS evaluation only, not used as optimization loss
# ------------------------------------------------------------

class LPIPSEvaluator(nn.Module):
    """
    Optional LPIPS evaluator used only for reporting.
    Inputs are expected in [0, 1].
    Internally converts to [-1, 1].
    """

    def __init__(self, net="alex"):
        super().__init__()

        if not HAS_LPIPS:
            raise ImportError("lpips package not installed. Run: pip install lpips")

        self.metric = lpips.LPIPS(net=net)

        for p in self.metric.parameters():
            p.requires_grad = False

        self.metric.eval()

    def forward(self, x0, x1):
        x0 = x0.clamp(0, 1) * 2.0 - 1.0
        x1 = x1.clamp(0, 1) * 2.0 - 1.0
        return self.metric(x0, x1).view(-1)


@torch.no_grad()
def compute_eval_metrics(x, target, lpips_alex=None, lpips_vgg=None):
    mse = F.mse_loss(x, target).item()
    l1 = F.l1_loss(x, target).item()
    psnr = psnr_from_mse(mse)

    out = {
        "mse": mse,
        "l1": l1,
        "psnr": psnr,
    }

    if lpips_alex is not None:
        out["lpips_alex"] = lpips_alex(x, target).mean().item()

    if lpips_vgg is not None:
        out["lpips_vgg"] = lpips_vgg(x, target).mean().item()

    return out


# ------------------------------------------------------------
# VGG paper-style feature losses
# ------------------------------------------------------------

FEATURE_POINTS = {
    # Same VGG-family extraction points as in arXiv:2302.04032, Table II.
    # The numbers mean "nth ReLU" in torchvision.features.
    "vgg11": [1, 2, 4, 8],
    "vgg16": [2, 4, 7, 13],
    "vgg16_bn": [2, 4, 7, 13],
    "vgg19": [2, 4, 8, 16],
}


def load_torchvision_vgg_features(arch):
    """
    Loads ImageNet-pretrained torchvision VGG features.

    Supports:
    - vgg11
    - vgg16
    - vgg16_bn
    - vgg19
    """

    if arch == "vgg11":
        factory = models.vgg11
        weights_name = "VGG11_Weights"
    elif arch == "vgg16":
        factory = models.vgg16
        weights_name = "VGG16_Weights"
    elif arch == "vgg16_bn":
        factory = models.vgg16_bn
        weights_name = "VGG16_BN_Weights"
    elif arch == "vgg19":
        factory = models.vgg19
        weights_name = "VGG19_Weights"
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    try:
        weights_enum = getattr(models, weights_name)
        weights = weights_enum.DEFAULT
        model = factory(weights=weights)
    except AttributeError:
        # Older torchvision fallback
        model = factory(pretrained=True)

    return model.features


def find_nth_relu_index(features, relu_ordinal):
    """
    Finds the torchvision.features index of the nth ReLU module.

    Example:
    In VGG16 without BN:
        2nd ReLU = relu1_2
        4th ReLU = relu2_2
        7th ReLU = relu3_3
        13th ReLU = relu5_3
    """

    count = 0

    for idx, layer in enumerate(features):
        if isinstance(layer, nn.ReLU):
            count += 1

            if count == relu_ordinal:
                return idx

    raise ValueError(
        f"Could not find ReLU number {relu_ordinal}. "
        f"Only found {count} ReLU layers."
    )


class VGGNthReLUFeatureLoss(nn.Module):
    """
    Single-layer VGG feature loss for direct image optimization.

    Input:
    - RGB images
    - shape [N, 3, H, W]
    - range [0, 1]

    Loss:
        mean((phi(x) - phi(target))^2)

    By default, features are used directly, following the "straightforward"
    extraction idea in arXiv:2302.04032.

    If --feature_norm is used, channel-wise feature normalization is applied,
    LPIPS-style. That is useful experimentally, but is not the default here.
    """

    def __init__(self, arch, relu_ordinal, feature_norm=False):
        super().__init__()

        self.arch = arch
        self.relu_ordinal = relu_ordinal
        self.feature_norm = feature_norm

        features = load_torchvision_vgg_features(arch)
        self.layer_index = find_nth_relu_index(features, relu_ordinal)

        # Keep only the network up to the requested ReLU.
        self.trunk = features[: self.layer_index + 1].eval()

        for p in self.trunk.parameters():
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
    def _channel_normalize(f, eps=1e-10):
        norm = torch.sqrt(torch.sum(f ** 2, dim=1, keepdim=True))
        return f / (norm + eps)

    def extract(self, x):
        x = x.clamp(0, 1)
        x = self._imagenet_norm(x)
        f = self.trunk(x)

        if self.feature_norm:
            f = self._channel_normalize(f)

        return f

    def distance_to_target_feature(self, x, target_feature):
        fx = self.extract(x)
        return torch.mean((fx - target_feature) ** 2)


def build_metric_specs(selected_metrics=None, selected_archs=None):
    """
    Metric names are:
        vgg11_relu01
        vgg11_relu02
        vgg11_relu04
        vgg11_relu08
        vgg16_relu02
        ...
    """

    specs = []

    if selected_archs is None:
        selected_archs = list(FEATURE_POINTS.keys())

    for arch in selected_archs:
        for relu_ordinal in FEATURE_POINTS[arch]:
            name = f"{arch}_relu{relu_ordinal:02d}"
            specs.append({
                "name": name,
                "arch": arch,
                "relu_ordinal": relu_ordinal,
            })

    if selected_metrics is not None:
        selected_metrics = set(selected_metrics)
        specs = [s for s in specs if s["name"] in selected_metrics]

        missing = selected_metrics - {s["name"] for s in specs}
        if len(missing) > 0:
            valid = [s["name"] for s in build_metric_specs()]
            raise ValueError(
                f"Unknown metric(s): {sorted(missing)}\n"
                f"Valid metrics are:\n" + "\n".join(valid)
            )

    return specs


# ------------------------------------------------------------
# Optimization
# ------------------------------------------------------------

def optimize_one_metric(
    metric_spec,
    input_img,
    target,
    outdir,
    device,
    steps=1000,
    lr=0.03,
    save_every=100,
    feature_norm=False,
    lpips_alex=None,
    lpips_vgg=None,
):
    metric_name = metric_spec["name"]
    arch = metric_spec["arch"]
    relu_ordinal = metric_spec["relu_ordinal"]

    outdir.mkdir(parents=True, exist_ok=True)

    metric = VGGNthReLUFeatureLoss(
        arch=arch,
        relu_ordinal=relu_ordinal,
        feature_norm=feature_norm,
    ).to(device)

    metric.eval()

    # Cache target feature once for speed.
    with torch.no_grad():
        target_feature = metric.extract(target).detach()

    x_start = input_img.clone().detach()
    x = x_start.clone().detach()
    x.requires_grad_(True)

    optimizer = torch.optim.Adam([x], lr=lr)

    history = []

    save_tensor_image(x_start, outdir / "start_input.png")
    save_tensor_image(target, outdir / "target.png")

    print("")
    print("=" * 80)
    print(f"Optimizing: {metric_name}")
    print(f"Architecture: {arch}")
    print(f"ReLU ordinal: {relu_ordinal}")
    print(f"Torchvision feature index: {metric.layer_index}")
    print("=" * 80)

    for step in range(steps + 1):
        optimizer.zero_grad(set_to_none=True)

        loss = metric.distance_to_target_feature(x, target_feature)
        loss.backward()

        grad_norm = x.grad.detach().norm().item()

        optimizer.step()

        # Valid image constraint.
        with torch.no_grad():
            x.clamp_(0, 1)

        loss_value = loss.item()

        if step % save_every == 0 or step == steps:
            eval_metrics = compute_eval_metrics(
                x.detach(),
                target,
                lpips_alex=lpips_alex,
                lpips_vgg=lpips_vgg,
            )

            row = {
                "step": step,
                "objective": loss_value,
                "grad_norm": grad_norm,
                **eval_metrics,
            }

            history.append(row)

            save_tensor_image(x, outdir / f"step_{step:05d}.png")

            msg = (
                f"[{metric_name:16s}] "
                f"step {step:5d} | "
                f"obj={loss_value:.6e} | "
                f"mse={eval_metrics['mse']:.6e} | "
                f"l1={eval_metrics['l1']:.6e} | "
                f"psnr={eval_metrics['psnr']:.2f} | "
                f"grad={grad_norm:.3e}"
            )

            if "lpips_alex" in eval_metrics:
                msg += f" | lpips_alex={eval_metrics['lpips_alex']:.6f}"

            if "lpips_vgg" in eval_metrics:
                msg += f" | lpips_vgg={eval_metrics['lpips_vgg']:.6f}"

            print(msg)

    final = x.detach().clamp(0, 1)

    save_tensor_image(final, outdir / "final.png")
    make_comparison_grid(x_start, target, final, outdir / "comparison_grid.png")

    # Save per-step history.
    history_path = outdir / "history.csv"
    with open(history_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    # Plot loss curve.
    steps_list = [h["step"] for h in history]
    obj_list = [h["objective"] for h in history]

    plt.figure()
    plt.plot(steps_list, obj_list)
    plt.xlabel("Step")
    plt.ylabel("Objective loss")
    plt.title(f"{metric_name} optimization from input")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outdir / "loss_curve.png", dpi=150)
    plt.close()

    final_metrics = compute_eval_metrics(
        final,
        target,
        lpips_alex=lpips_alex,
        lpips_vgg=lpips_vgg,
    )

    summary = {
        "metric": metric_name,
        "architecture": arch,
        "relu_ordinal": relu_ordinal,
        "torchvision_layer_index": metric.layer_index,
        "feature_norm": feature_norm,
        "final_objective": history[-1]["objective"],
        "final_grad_norm": history[-1]["grad_norm"],
        **final_metrics,
        "outdir": str(outdir),
    }

    del metric

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Direct image optimization using paper-style VGG loss networks."
    )

    parser.add_argument("--target", type=str, required=True, help="Target image path.")
    parser.add_argument("--input", type=str, required=True, help="Input image used as initialization.")

    parser.add_argument("--resize", type=int, default=256)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--outdir", type=str, default="direct_optimization_paper_vgg")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "--archs",
        nargs="+",
        default=["vgg11", "vgg16", "vgg16_bn", "vgg19"],
        choices=["vgg11", "vgg16", "vgg16_bn", "vgg19"],
        help="Architectures to test.",
    )

    parser.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help=(
            "Optional subset of metrics to run, e.g. "
            "vgg11_relu01 vgg16_relu02 vgg19_relu16. "
            "If omitted, all selected architectures/layers are run."
        ),
    )

    parser.add_argument(
        "--feature_norm",
        action="store_true",
        help=(
            "Apply LPIPS-style channel normalization to feature maps before distance. "
            "Default is False, closer to straightforward raw activation matching."
        ),
    )

    parser.add_argument(
        "--eval_lpips",
        action="store_true",
        help="Also report LPIPS-Alex and LPIPS-VGG for the optimized image.",
    )

    parser.add_argument(
        "--list_metrics",
        action="store_true",
        help="Print all valid metric names and exit.",
    )

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available. Falling back to CPU.")
        args.device = "cpu"

    device = torch.device(args.device)

    if args.list_metrics:
        for spec in build_metric_specs():
            print(spec["name"])
        return

    metric_specs = build_metric_specs(
        selected_metrics=args.metrics,
        selected_archs=args.archs,
    )

    root_outdir = Path(args.outdir)
    root_outdir.mkdir(parents=True, exist_ok=True)

    target = load_image(args.target, resize=args.resize, device=device)
    input_img = load_image(args.input, resize=args.resize, device=device)

    lpips_alex = None
    lpips_vgg = None

    if args.eval_lpips:
        if not HAS_LPIPS:
            raise ImportError(
                "--eval_lpips was used, but lpips is not installed. "
                "Run: pip install lpips"
            )

        lpips_alex = LPIPSEvaluator(net="alex").to(device).eval()
        lpips_vgg = LPIPSEvaluator(net="vgg").to(device).eval()

    summary_rows = []

    for spec in metric_specs:
        metric_outdir = root_outdir / spec["name"]

        row = optimize_one_metric(
            metric_spec=spec,
            input_img=input_img,
            target=target,
            outdir=metric_outdir,
            device=device,
            steps=args.steps,
            lr=args.lr,
            save_every=args.save_every,
            feature_norm=args.feature_norm,
            lpips_alex=lpips_alex,
            lpips_vgg=lpips_vgg,
        )

        summary_rows.append(row)

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