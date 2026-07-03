import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

try:
    import lpips
except ImportError:
    lpips = None


IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".bmp"]


# -----------------------------------------------------------------------------
# BAPPS datasets
# -----------------------------------------------------------------------------

def list_images(folder):
    folder = Path(folder)
    files = []
    for ext in IMAGE_EXTS:
        files.extend(folder.glob(f"*{ext}"))
    return sorted(files)


def list_npy(folder):
    return sorted(Path(folder).glob("*.npy"))


class BAPPS2AFCDataset(Dataset):
    """
    BAPPS 2AFC subset.

    Expected structure:
        root/ref/
        root/p0/
        root/p1/
        root/judge/

    The judge value is in [0, 1]:
        closer to 0 means humans preferred p0;
        closer to 1 means humans preferred p1.
    """

    def __init__(self, root, load_size=64):
        self.root = Path(root)
        self.ref_paths = list_images(self.root / "ref")
        self.p0_paths = list_images(self.root / "p0")
        self.p1_paths = list_images(self.root / "p1")
        self.judge_paths = list_npy(self.root / "judge")

        assert len(self.ref_paths) > 0, f"No images found in {self.root / 'ref'}"
        assert len(self.ref_paths) == len(self.p0_paths) == len(self.p1_paths) == len(self.judge_paths), (
            f"Mismatch in {root}: "
            f"ref={len(self.ref_paths)}, p0={len(self.p0_paths)}, "
            f"p1={len(self.p1_paths)}, judge={len(self.judge_paths)}"
        )

        self.tf = transforms.Compose([
            transforms.Resize((load_size, load_size)),
            transforms.ToTensor(),  # [0, 1]
        ])

    def __len__(self):
        return len(self.ref_paths)

    def __getitem__(self, idx):
        ref = self.tf(Image.open(self.ref_paths[idx]).convert("RGB"))
        p0 = self.tf(Image.open(self.p0_paths[idx]).convert("RGB"))
        p1 = self.tf(Image.open(self.p1_paths[idx]).convert("RGB"))
        judge = float(np.load(self.judge_paths[idx]).reshape(-1)[0])
        return ref, p0, p1, torch.tensor(judge, dtype=torch.float32)


class BAPPSJNDDataset(Dataset):
    """
    BAPPS JND subset.

    Expected structure:
        root/p0/
        root/p1/
        root/same/
    """

    def __init__(self, root, load_size=64):
        self.root = Path(root)
        self.p0_paths = list_images(self.root / "p0")
        self.p1_paths = list_images(self.root / "p1")
        self.same_paths = list_npy(self.root / "same")

        assert len(self.p0_paths) > 0, f"No images found in {self.root / 'p0'}"
        assert len(self.p0_paths) == len(self.p1_paths) == len(self.same_paths), (
            f"Mismatch in {root}: "
            f"p0={len(self.p0_paths)}, p1={len(self.p1_paths)}, same={len(self.same_paths)}"
        )

        self.tf = transforms.Compose([
            transforms.Resize((load_size, load_size)),
            transforms.ToTensor(),  # [0, 1]
        ])

    def __len__(self):
        return len(self.p0_paths)

    def __getitem__(self, idx):
        p0 = self.tf(Image.open(self.p0_paths[idx]).convert("RGB"))
        p1 = self.tf(Image.open(self.p1_paths[idx]).convert("RGB"))
        same = float(np.load(self.same_paths[idx]).reshape(-1)[0])
        return p0, p1, torch.tensor(same, dtype=torch.float32)


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

VGG_ARCH_SPECS = {
    # Feature extraction points chosen to match the table from Pihlgren et al.
    # The numbers are counted as the 1st ReLU, 2nd ReLU, etc. in torchvision's
    # features Sequential module.
    "vgg11": {
        "factory": "vgg11",
        "weights_enum": "VGG11_Weights",
        "relu_points": [1, 2, 4, 8],
    },
    "vgg16": {
        "factory": "vgg16",
        "weights_enum": "VGG16_Weights",
        "relu_points": [2, 4, 7, 13],
    },
    "vgg16_bn": {
        "factory": "vgg16_bn",
        "weights_enum": "VGG16_BN_Weights",
        "relu_points": [2, 4, 7, 13],
    },
    "vgg19": {
        "factory": "vgg19",
        "weights_enum": "VGG19_Weights",
        "relu_points": [2, 4, 8, 16],
    },
}


class VGGArchMetricBank(nn.Module):
    """
    One frozen VGG architecture that returns multiple feature distances.

    For one architecture, this module computes distances for:
        - each selected ReLU extraction point individually;
        - optionally, the mean over the four selected ReLU distances.

    Input tensors must be RGB images in [0, 1], shape [N, 3, H, W].
    Internally, ImageNet normalization is applied before VGG feature extraction.
    """

    def __init__(self, architecture, pretrained=True, normalize_features=True, include_multi=True):
        super().__init__()

        if architecture not in VGG_ARCH_SPECS:
            raise ValueError(f"Unknown architecture: {architecture}")

        self.architecture = architecture
        self.pretrained = pretrained
        self.normalize_features = normalize_features
        self.include_multi = include_multi
        self.relu_points = VGG_ARCH_SPECS[architecture]["relu_points"]

        features = self._load_features(architecture=architecture, pretrained=pretrained)
        self.relu_to_feature_index = self._relu_number_to_feature_index(features)

        missing = [r for r in self.relu_points if r not in self.relu_to_feature_index]
        if missing:
            raise RuntimeError(
                f"Architecture {architecture} does not contain requested ReLU numbers {missing}. "
                f"Detected ReLUs: {sorted(self.relu_to_feature_index.keys())}"
            )

        self.selected_feature_indices = {
            r: self.relu_to_feature_index[r] for r in self.relu_points
        }
        self.max_feature_index = max(self.selected_feature_indices.values())
        self.features = features[: self.max_feature_index + 1].eval()

        for p in self.features.parameters():
            p.requires_grad = False

        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

    @staticmethod
    def _load_features(architecture, pretrained=True):
        factory_name = VGG_ARCH_SPECS[architecture]["factory"]
        weights_enum_name = VGG_ARCH_SPECS[architecture]["weights_enum"]
        factory = getattr(models, factory_name)

        if not pretrained:
            try:
                return factory(weights=None).features
            except TypeError:
                return factory(pretrained=False).features

        # New torchvision API: models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        weights_enum = getattr(models, weights_enum_name, None)
        if weights_enum is not None:
            weights = weights_enum.IMAGENET1K_V1
            return factory(weights=weights).features

        # Old torchvision API fallback.
        return factory(pretrained=True).features

    @staticmethod
    def _relu_number_to_feature_index(features):
        mapping = {}
        relu_count = 0
        for idx, layer in enumerate(features):
            if isinstance(layer, nn.ReLU):
                relu_count += 1
                mapping[relu_count] = idx
        return mapping

    def _imagenet_norm(self, x):
        return (x - self.mean) / self.std

    @staticmethod
    def _normalize_feat(f, eps=1e-10):
        norm = torch.sqrt(torch.sum(f ** 2, dim=1, keepdim=True))
        return f / (norm + eps)

    def _extract_selected_features(self, x):
        x = self._imagenet_norm(x)
        feats = {}
        wanted = set(self.selected_feature_indices.values())

        for idx, layer in enumerate(self.features):
            x = layer(x)
            if idx in wanted:
                feats[idx] = x

        return feats

    @staticmethod
    def _mse_per_image(a, b):
        return ((a - b) ** 2).mean(dim=[1, 2, 3])

    def _candidate_name(self, relu_number):
        return f"{self.architecture}_relu{relu_number}"

    def _multi_candidate_name(self):
        joined = "_".join(str(r) for r in self.relu_points)
        return f"{self.architecture}_multi_relu{joined}"

    def candidate_metadata(self):
        rows = []
        for relu_number in self.relu_points:
            rows.append({
                "candidate": self._candidate_name(relu_number),
                "architecture": self.architecture,
                "candidate_type": "single_relu",
                "relu_number": relu_number,
                "relu_points": str([relu_number]),
                "torchvision_feature_index": self.selected_feature_indices[relu_number],
                "pretrained": self.pretrained,
                "normalize_features": self.normalize_features,
            })

        if self.include_multi:
            rows.append({
                "candidate": self._multi_candidate_name(),
                "architecture": self.architecture,
                "candidate_type": "multi_relu_mean",
                "relu_number": np.nan,
                "relu_points": str(self.relu_points),
                "torchvision_feature_index": str([
                    self.selected_feature_indices[r] for r in self.relu_points
                ]),
                "pretrained": self.pretrained,
                "normalize_features": self.normalize_features,
            })

        return rows

    def forward(self, x0, x1):
        f0 = self._extract_selected_features(x0)
        f1 = self._extract_selected_features(x1)

        out = {}
        single_distances = []

        for relu_number in self.relu_points:
            idx = self.selected_feature_indices[relu_number]
            a = f0[idx]
            b = f1[idx]

            if self.normalize_features:
                a = self._normalize_feat(a)
                b = self._normalize_feat(b)

            d = self._mse_per_image(a, b)
            out[self._candidate_name(relu_number)] = d
            single_distances.append(d)

        if self.include_multi:
            out[self._multi_candidate_name()] = torch.stack(single_distances, dim=0).mean(dim=0)

        return out


class LPIPSVGGDistance(nn.Module):
    """
    Optional official learned LPIPS-VGG baseline.
    Input tensors are [0, 1]; LPIPS expects [-1, 1].
    """

    def __init__(self):
        super().__init__()
        if lpips is None:
            raise ImportError("lpips is not installed. Run: pip install lpips")
        self.metric = lpips.LPIPS(net="vgg")

    def forward(self, x0, x1):
        x0 = x0 * 2.0 - 1.0
        x1 = x1 * 2.0 - 1.0
        return {"lpips_vgg": self.metric(x0, x1).view(-1)}

    def candidate_metadata(self):
        return [{
            "candidate": "lpips_vgg",
            "architecture": "lpips_vgg",
            "candidate_type": "learned_lpips_baseline",
            "relu_number": np.nan,
            "relu_points": "official_lpips_vgg",
            "torchvision_feature_index": "official_lpips_vgg",
            "pretrained": True,
            "normalize_features": True,
        }]


# -----------------------------------------------------------------------------
# Evaluation helpers
# -----------------------------------------------------------------------------

def voc_ap(rec, prec):
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))

    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    i = np.where(mrec[1:] != mrec[:-1])[0]
    return np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])


def make_loader(dataset, batch_size, num_workers, device):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )


@torch.no_grad()
def eval_2afc_metric_bank(metric_bank, root, batch_size, device, num_workers, load_size):
    ds = BAPPS2AFCDataset(root, load_size=load_size)
    loader = make_loader(ds, batch_size=batch_size, num_workers=num_workers, device=device)

    score_lists = None

    for ref, p0, p1, judge in tqdm(loader, desc=f"2AFC {root}"):
        ref = ref.to(device, non_blocking=True)
        p0 = p0.to(device, non_blocking=True)
        p1 = p1.to(device, non_blocking=True)

        d0_dict = metric_bank(ref, p0)
        d1_dict = metric_bank(ref, p1)
        judge_np = judge.cpu().numpy().reshape(-1)

        if score_lists is None:
            score_lists = {name: [] for name in d0_dict.keys()}

        for name in d0_dict.keys():
            d0 = d0_dict[name].detach().cpu().numpy().reshape(-1)
            d1 = d1_dict[name].detach().cpu().numpy().reshape(-1)
            scores = (
                (d0 < d1) * (1.0 - judge_np)
                + (d1 < d0) * judge_np
                + (d1 == d0) * 0.5
            )
            score_lists[name].extend(scores.tolist())

    return {name: float(np.mean(values)) for name, values in score_lists.items()}


@torch.no_grad()
def eval_jnd_metric_bank(metric_bank, root, batch_size, device, num_workers, load_size):
    ds = BAPPSJNDDataset(root, load_size=load_size)
    loader = make_loader(ds, batch_size=batch_size, num_workers=num_workers, device=device)

    dscores = None
    sames_all = []

    for p0, p1, same in tqdm(loader, desc=f"JND {root}"):
        p0 = p0.to(device, non_blocking=True)
        p1 = p1.to(device, non_blocking=True)

        d_dict = metric_bank(p0, p1)

        if dscores is None:
            dscores = {name: [] for name in d_dict.keys()}

        for name, d in d_dict.items():
            dscores[name].extend(d.detach().cpu().numpy().reshape(-1).tolist())

        sames_all.extend(same.cpu().numpy().reshape(-1).tolist())

    sames = np.array(sames_all)
    out = {}

    for name, values in dscores.items():
        d = np.array(values)
        sorted_inds = np.argsort(d)
        sames_sorted = sames[sorted_inds]

        tps = np.cumsum(sames_sorted)
        fps = np.cumsum(1.0 - sames_sorted)
        fns = np.sum(sames_sorted) - tps

        prec = tps / np.maximum(tps + fps, 1e-12)
        rec = tps / np.maximum(tps + fns, 1e-12)
        out[name] = float(voc_ap(rec, prec))

    return out


def rows_from_scores(scores, metadata_by_candidate, mode, subset):
    rows = []
    for candidate, score in scores.items():
        meta = metadata_by_candidate[candidate]
        row = dict(meta)
        row.update({
            "mode": mode,
            "subset": subset,
            "score": score,
            "score_percent": 100.0 * score,
        })
        rows.append(row)
    return rows


def make_summary(df, summary_output):
    # Pivot to one row per candidate, keeping all subset scores as columns.
    id_cols = [
        "candidate",
        "architecture",
        "candidate_type",
        "relu_number",
        "relu_points",
        "torchvision_feature_index",
        "pretrained",
        "normalize_features",
    ]

    pivot = df.pivot_table(
        index=id_cols,
        columns=["mode", "subset"],
        values="score_percent",
        aggfunc="mean",
    )
    pivot.columns = [f"{mode}_{subset.replace('/', '_')}" for mode, subset in pivot.columns]
    pivot = pivot.reset_index()

    score_cols = [c for c in pivot.columns if c.startswith("2afc_") or c.startswith("jnd_")]
    twoafc_cols = [c for c in score_cols if c.startswith("2afc_")]
    jnd_cols = [c for c in score_cols if c.startswith("jnd_")]

    pivot["mean_2afc"] = pivot[twoafc_cols].mean(axis=1) if twoafc_cols else np.nan
    pivot["mean_jnd"] = pivot[jnd_cols].mean(axis=1) if jnd_cols else np.nan
    pivot["mean_all"] = pivot[score_cols].mean(axis=1) if score_cols else np.nan

    # A project-specific summary useful for tone mapping: emphasize color and
    # traditional distortions more than CNN artifacts.
    priority_cols = [
        "2afc_val_color",
        "2afc_val_traditional",
        "jnd_val_traditional",
    ]
    existing_priority_cols = [c for c in priority_cols if c in pivot.columns]
    pivot["mean_tonemapping_priority"] = (
        pivot[existing_priority_cols].mean(axis=1) if existing_priority_cols else np.nan
    )

    sort_col = "mean_tonemapping_priority" if "mean_tonemapping_priority" in pivot.columns else "mean_all"
    pivot = pivot.sort_values(sort_col, ascending=False)
    pivot.to_csv(summary_output, index=False)
    return pivot


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate VGG architecture/layer perceptual distances on BAPPS validation subsets."
    )
    parser.add_argument("--root", type=str, default="dataset")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--load_size", type=int, default=64)
    parser.add_argument("--output", type=str, default="bapps_vgg_arch_layers_results.csv")
    parser.add_argument("--summary_output", type=str, default=None)

    parser.add_argument(
        "--architectures",
        nargs="+",
        default=["vgg11", "vgg16", "vgg16_bn", "vgg19"],
        choices=list(VGG_ARCH_SPECS.keys()),
    )
    parser.add_argument(
        "--twoafc_sets",
        nargs="+",
        default=["val/traditional", "val/cnn", "val/color"],
    )
    parser.add_argument(
        "--jnd_sets",
        nargs="+",
        default=["val/traditional", "val/cnn"],
    )
    parser.add_argument(
        "--include_multi",
        action="store_true",
        default=True,
        help="Also evaluate the mean over the four selected ReLU distances for each architecture.",
    )
    parser.add_argument(
        "--no_multi",
        dest="include_multi",
        action="store_false",
        help="Disable multi-ReLU mean candidates.",
    )
    parser.add_argument(
        "--include_lpips_vgg",
        action="store_true",
        help="Also evaluate the official learned LPIPS-VGG baseline.",
    )
    parser.add_argument(
        "--no_channel_norm",
        dest="normalize_features",
        action="store_false",
        help="Disable LPIPS-style channel normalization before feature MSE.",
    )
    parser.set_defaults(normalize_features=True)
    parser.add_argument(
        "--random_weights",
        action="store_true",
        help="Use random VGG weights instead of ImageNet pretrained weights.",
    )

    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available. Falling back to CPU.")
        args.device = "cpu"

    device = torch.device(args.device)
    root = Path(args.root)
    pretrained = not args.random_weights

    if args.summary_output is None:
        output_path = Path(args.output)
        args.summary_output = str(output_path.with_name(output_path.stem + "_summary.csv"))

    rows = []

    for architecture in args.architectures:
        print("\n" + "=" * 88)
        print(f"Evaluating architecture: {architecture}")
        print(f"ReLU extraction points: {VGG_ARCH_SPECS[architecture]['relu_points']}")
        print("=" * 88)

        metric_bank = VGGArchMetricBank(
            architecture=architecture,
            pretrained=pretrained,
            normalize_features=args.normalize_features,
            include_multi=args.include_multi,
        ).to(device).eval()

        metadata_by_candidate = {
            row["candidate"]: row for row in metric_bank.candidate_metadata()
        }

        for subset in args.twoafc_sets:
            subset_root = root / "2afc" / subset
            scores = eval_2afc_metric_bank(
                metric_bank=metric_bank,
                root=subset_root,
                batch_size=args.batch_size,
                device=device,
                num_workers=args.num_workers,
                load_size=args.load_size,
            )
            rows.extend(rows_from_scores(scores, metadata_by_candidate, mode="2afc", subset=subset))
            for name, score in scores.items():
                print(f"{name:<32} | 2AFC | {subset:<15}: {100.0 * score:.2f}")

        for subset in args.jnd_sets:
            subset_root = root / "jnd" / subset
            scores = eval_jnd_metric_bank(
                metric_bank=metric_bank,
                root=subset_root,
                batch_size=args.batch_size,
                device=device,
                num_workers=args.num_workers,
                load_size=args.load_size,
            )
            rows.extend(rows_from_scores(scores, metadata_by_candidate, mode="jnd", subset=subset))
            for name, score in scores.items():
                print(f"{name:<32} | JND  | {subset:<15}: {100.0 * score:.2f}")

        del metric_bank
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.include_lpips_vgg:
        print("\n" + "=" * 88)
        print("Evaluating optional baseline: lpips_vgg")
        print("=" * 88)

        metric_bank = LPIPSVGGDistance().to(device).eval()
        metadata_by_candidate = {
            row["candidate"]: row for row in metric_bank.candidate_metadata()
        }

        for subset in args.twoafc_sets:
            subset_root = root / "2afc" / subset
            scores = eval_2afc_metric_bank(
                metric_bank=metric_bank,
                root=subset_root,
                batch_size=args.batch_size,
                device=device,
                num_workers=args.num_workers,
                load_size=args.load_size,
            )
            rows.extend(rows_from_scores(scores, metadata_by_candidate, mode="2afc", subset=subset))
            print(f"lpips_vgg{'':<23} | 2AFC | {subset:<15}: {100.0 * scores['lpips_vgg']:.2f}")

        for subset in args.jnd_sets:
            subset_root = root / "jnd" / subset
            scores = eval_jnd_metric_bank(
                metric_bank=metric_bank,
                root=subset_root,
                batch_size=args.batch_size,
                device=device,
                num_workers=args.num_workers,
                load_size=args.load_size,
            )
            rows.extend(rows_from_scores(scores, metadata_by_candidate, mode="jnd", subset=subset))
            print(f"lpips_vgg{'':<23} | JND  | {subset:<15}: {100.0 * scores['lpips_vgg']:.2f}")

        del metric_bank
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)
    summary = make_summary(df, args.summary_output)

    print("\nSaved detailed results to:", args.output)
    print("Saved summary results to: ", args.summary_output)
    print("\nTop candidates by mean_tonemapping_priority:")
    display_cols = [
        "candidate",
        "architecture",
        "candidate_type",
        "relu_points",
        "mean_tonemapping_priority",
        "mean_2afc",
        "mean_jnd",
        "mean_all",
    ]
    display_cols = [c for c in display_cols if c in summary.columns]
    print(summary[display_cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
