import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

import lpips


IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".bmp"]


def list_images(folder):
    folder = Path(folder)
    files = []
    for ext in IMAGE_EXTS:
        files.extend(folder.glob(f"*{ext}"))
    return sorted(files)


def list_npy(folder):
    return sorted(Path(folder).glob("*.npy"))


class BAPPS2AFCDataset(Dataset):
    def __init__(self, root):
        self.root = Path(root)
        self.ref_paths = list_images(self.root / "ref")
        self.p0_paths = list_images(self.root / "p0")
        self.p1_paths = list_images(self.root / "p1")
        self.judge_paths = list_npy(self.root / "judge")

        assert len(self.ref_paths) == len(self.p0_paths) == len(self.p1_paths) == len(self.judge_paths), \
            f"Mismatch in {root}"

        self.tf = transforms.Compose([
            transforms.Resize((64, 64)),
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
    def __init__(self, root):
        self.root = Path(root)
        self.p0_paths = list_images(self.root / "p0")
        self.p1_paths = list_images(self.root / "p1")
        self.same_paths = list_npy(self.root / "same")

        assert len(self.p0_paths) == len(self.p1_paths) == len(self.same_paths), \
            f"Mismatch in {root}"

        self.tf = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),  # [0, 1]
        ])

    def __len__(self):
        return len(self.p0_paths)

    def __getitem__(self, idx):
        p0 = self.tf(Image.open(self.p0_paths[idx]).convert("RGB"))
        p1 = self.tf(Image.open(self.p1_paths[idx]).convert("RGB"))
        same = float(np.load(self.same_paths[idx]).reshape(-1)[0])
        return p0, p1, torch.tensor(same, dtype=torch.float32)


def voc_ap(rec, prec):
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))

    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    i = np.where(mrec[1:] != mrec[:-1])[0]
    return np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])


class VGGFeatureDistance(nn.Module):
    """
    VGG16 feature distance for BAPPS patches.

    Input tensors are expected in [0, 1], shape [N, 3, H, W].
    Features are ImageNet-normalized internally.
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
        "vgg_r22": ["relu2_2"],
        "vgg_r33": ["relu3_3"],
        "vgg_r43": ["relu4_3"],
        "vgg_multi_123": ["relu1_2", "relu2_2", "relu3_3"],
        "vgg_multi_1234": ["relu1_2", "relu2_2", "relu3_3", "relu4_3"],
    }

    def __init__(self, candidate_name, normalize_features=True):
        super().__init__()

        if candidate_name not in self.CANDIDATES:
            raise ValueError(f"Unknown candidate {candidate_name}")

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

    def _extract(self, x):
        feats = {}
        x = self._imagenet_norm(x)

        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in self.layer_ids:
                feats[i] = x

        return feats

    @staticmethod
    def _normalize_feat(f, eps=1e-10):
        norm = torch.sqrt(torch.sum(f ** 2, dim=1, keepdim=True))
        return f / (norm + eps)

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
            d = d.mean(dim=[1, 2, 3])  # one distance per image
            total = total + d

        return total / len(self.layer_ids)


class LPIPSVGGDistance(nn.Module):
    def __init__(self):
        super().__init__()
        self.metric = lpips.LPIPS(net="vgg")

    def forward(self, x0, x1):
        # LPIPS expects [-1, 1]
        x0 = x0 * 2.0 - 1.0
        x1 = x1 * 2.0 - 1.0
        return self.metric(x0, x1).view(-1)


def make_metric(name, device):
    if name == "lpips_vgg":
        metric = LPIPSVGGDistance()
    else:
        metric = VGGFeatureDistance(name, normalize_features=True)

    metric = metric.to(device)
    metric.eval()
    return metric


@torch.no_grad()
def eval_2afc(metric, root, batch_size, device, num_workers):
    ds = BAPPS2AFCDataset(root)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    d0s, d1s, gts = [], [], []

    for ref, p0, p1, judge in tqdm(loader, desc=f"2AFC {root}"):
        ref = ref.to(device)
        p0 = p0.to(device)
        p1 = p1.to(device)

        d0 = metric(ref, p0).detach().cpu().numpy().reshape(-1)
        d1 = metric(ref, p1).detach().cpu().numpy().reshape(-1)

        d0s.extend(d0.tolist())
        d1s.extend(d1.tolist())
        gts.extend(judge.numpy().reshape(-1).tolist())

    d0s = np.array(d0s)
    d1s = np.array(d1s)
    gts = np.array(gts)

    scores = (d0s < d1s) * (1.0 - gts) + (d1s < d0s) * gts + (d1s == d0s) * 0.5
    return float(np.mean(scores))


@torch.no_grad()
def eval_jnd(metric, root, batch_size, device, num_workers):
    ds = BAPPSJNDDataset(root)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    dscores, sames = [], []

    for p0, p1, same in tqdm(loader, desc=f"JND {root}"):
        p0 = p0.to(device)
        p1 = p1.to(device)

        d = metric(p0, p1).detach().cpu().numpy().reshape(-1)

        dscores.extend(d.tolist())
        sames.extend(same.numpy().reshape(-1).tolist())

    dscores = np.array(dscores)
    sames = np.array(sames)

    sorted_inds = np.argsort(dscores)
    sames_sorted = sames[sorted_inds]

    tps = np.cumsum(sames_sorted)
    fps = np.cumsum(1.0 - sames_sorted)
    fns = np.sum(sames_sorted) - tps

    prec = tps / np.maximum(tps + fps, 1e-12)
    rec = tps / np.maximum(tps + fns, 1e-12)

    return float(voc_ap(rec, prec))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="dataset")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default="bapps_6_vgg_results.csv")
    parser.add_argument(
        "--candidates",
        nargs="+",
        default=[
            "vgg_r12",
            "vgg_r22",
            "vgg_r33",
            "vgg_r43",
            "vgg_multi_123",
            "lpips_vgg",
        ],
    )
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available. Falling back to CPU.")
        args.device = "cpu"

    device = torch.device(args.device)

    twoafc_sets = ["val/traditional", "val/cnn", "val/color"]
    jnd_sets = ["val/traditional", "val/cnn"]

    rows = []

    for candidate in args.candidates:
        print(f"\nEvaluating candidate: {candidate}")
        metric = make_metric(candidate, device)

        for subset in twoafc_sets:
            subset_root = Path(args.root) / "2afc" / subset
            score = eval_2afc(metric, subset_root, args.batch_size, device, args.num_workers)
            rows.append({
                "candidate": candidate,
                "mode": "2afc",
                "subset": subset,
                "score": score,
                "score_percent": 100.0 * score,
            })
            print(f"{candidate} | 2AFC | {subset}: {100.0 * score:.2f}")

        for subset in jnd_sets:
            subset_root = Path(args.root) / "jnd" / subset
            score = eval_jnd(metric, subset_root, args.batch_size, device, args.num_workers)
            rows.append({
                "candidate": candidate,
                "mode": "jnd",
                "subset": subset,
                "score": score,
                "score_percent": 100.0 * score,
            })
            print(f"{candidate} | JND  | {subset}: {100.0 * score:.2f}")

        del metric
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)

    print("\nSaved results to:", args.output)
    print(df)


if __name__ == "__main__":
    main()