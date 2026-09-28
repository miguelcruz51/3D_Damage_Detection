# 3D Damage Detection on Metallic Components

Deep-learning pipeline for detecting and localizing surface damage on metallic engineering
components from 3D geometry. The system takes a raw component mesh, segments its surface
point-by-point into *healthy* and *damaged*, and localizes damage as bounding-box regions.

---

## Project Overview

Given the 3D scan (STL mesh) of a metallic part, the pipeline predicts which surface points
are damaged and groups them into bounding-box damage regions. It compares seven point-cloud
architectures and seven loss functions to find the configuration that best handles the
severe class imbalance inherent to damage detection.

## Engineering Problem

Surface damage on a metallic component covers only **~1.5–2% of its surface**. Under this
extreme class imbalance, a naive classifier reaches high accuracy while missing the damage
that actually matters. The engineering challenge is to reliably segment this rare damaged
class, which requires imbalance-aware loss functions and a region-based evaluation that goes
beyond per-point accuracy. Reliable, automated localization is a prerequisite for automated
repair of metallic components.

## Research Pipeline

```
STL mesh
   │  load + uniform surface sampling (Open3D), estimate normals
   ▼
Point cloud (XYZ + normals)
   │  grid-based spatial partition into fixed-size local patches
   ▼
Local patches
   │  pc_normalize (center on centroid, scale to unit sphere)
   ▼
Normalized patch  [B, C, N]
   │  point-cloud segmentation model
   ▼
Per-point log-probabilities  [B, N, 2]  (0 = healthy, 1 = damaged)
   │  stochastic voting + threshold, stitch patches back
   ▼
Per-point damage segmentation
   │  DBSCAN clustering → axis-aligned bounding boxes
   ▼
Bounding-box damage localization
```

See [`docs/methodology.md`](docs/methodology.md) for the full description.

## Implemented Models

| Model | Type |
|---|---|
| PointNet | Global-feature MLP over the point set |
| PointNet++ (SSG) | Hierarchical set abstraction |
| DGCNN | Dynamic edge-convolution over k-NN graphs |
| KPConv (v2) | Kernel point convolution |
| Point Transformer (v3) | Self-attention over local neighborhoods |
| PointNet + Transfer | PointNet with a pre-trained encoder |
| PointNet++ + Transfer | PointNet++ with a pre-trained encoder |

All models output per-point log-probabilities of shape `[B, N, 2]`. See
[`src/models/README.md`](src/models/README.md) for the comparison table.

## Dataset

- **16 component types** (`Component1`–`Component16`).
- **NPZ patches**, each with `points (N,3)`, `normals (N,3)`, and `labels (N,)`
  (0 = healthy, 1 = damaged).
- **Train / validation / test split** via ShuffledFileList JSON files under
  `data/damaged_components/train_test_split/`.

The dataset and trained weights are large and are not committed to the repository (see
[`.gitignore`](.gitignore)). Place the dataset under `data/damaged_components/` following the
structure in [`docs/methodology.md`](docs/methodology.md). If the data is not present,
`DamageDataset` raises a `FileNotFoundError` explaining how to provide it.

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

A CUDA-capable GPU is recommended for training (the study used an NVIDIA RTX 2080, 8 GB
VRAM).

## Usage

**Train** a model from a YAML config:

```bash
python scripts/train.py --config configs/pointnet2_boundary_aware.yaml
```

**Evaluate** a trained run (per-point, plus optional bounding-box evaluation):

```bash
python scripts/evaluate.py --log_dir logs/<experiment> --macro-eval
```

**Run inference** on a raw STL file:

```bash
python scripts/inference.py \
    --stl_path path/to/component.STL \
    --model_path path/to/best_model.pth \
    --output_path path/to/output \
    --npoints 2048 --threshold 0.5 --n_votes 6
```

With `--output_path`, predictions are saved as `.npz` (per-point probabilities) and `.txt`
(binary predictions); without it, an interactive viewer is launched.

## Key Results

Best overall: **PointNet++ (SSG) + boundary_aware loss**.

| Metric | Value |
|---|---|
| PR-AUC | **0.9201** |
| F1-Score | 0.8543 |
| mIoU | 0.8358 |

Full results across the 49-experiment grid are in [`results/`](results/README.md) and
[`experiments/`](experiments/README.md).

## Repository Structure

```
clean_repo/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── configs/            # YAML experiment configs
├── src/
│   ├── data/           # DamageDataset, preprocessing, augmentation
│   ├── models/         # architectures + losses
│   ├── training/       # training loop
│   ├── evaluation/     # metrics, bbox eval, inference, visualization
│   └── utils/          # checkpoint utilities
├── scripts/            # train / evaluate / inference entry points
├── experiments/        # experiment grid documentation
├── results/            # results summary + metrics CSV
├── docs/               # audit + methodology
└── tests/              # unit and property-based tests
```

## Research Context

This work targets **automated repair of metallic components**: reliable, automated
localization of surface damage is the first step toward robotic or automated repair
workflows, replacing slow and subjective manual inspection.

## Acknowledgements

This project started from
[Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch) by Xu Yan.
The PointNet and PointNet++ implementations, the shared set-abstraction / feature-propagation
layers (`src/models/pointnet2_utils.py`, `src/models/pointnet_utils.py`), the point-cloud
augmentation functions, and the overall training-script structure derive from that repository
and were adapted here for binary damage segmentation. Full credit to the original authors.

If you use this repository, please also cite the upstream work:

```bibtex
@article{Pytorch_Pointnet_Pointnet2,
  Author  = {Xu Yan},
  Title   = {Pointnet/Pointnet++ Pytorch},
  Journal = {https://github.com/yanx27/Pointnet_Pointnet2_pytorch},
  Year    = {2019}
}

@InProceedings{yan2020pointasnl,
  title   = {PointASNL: Robust Point Clouds Processing using Nonlocal Neural Networks with Adaptive Sampling},
  author  = {Yan, Xu and Zheng, Chaoda and Li, Zhen and Wang, Sheng and Cui, Shuguang},
  journal = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year    = {2020}
}

@InProceedings{yan2021sparse,
  title   = {Sparse Single Sweep LiDAR Point Cloud Segmentation via Learning Contextual Shape Priors from Scene Completion},
  author  = {Yan, Xu and Gao, Jiantao and Li, Jie and Zhang, Ruimao and Li, Zhen and Huang, Rui and Cui, Shuguang},
  journal = {AAAI Conference on Artificial Intelligence ({AAAI})},
  year    = {2021}
}

@InProceedings{yan20222dpass,
  title   = {2DPASS: 2D Priors Assisted Semantic Segmentation on LiDAR Point Clouds},
  author  = {Xu Yan and Jiantao Gao and Chaoda Zheng and Chao Zheng and Ruimao Zhang and Shuguang Cui and Zhen Li},
  journal = {ECCV},
  year    = {2022}
}
```

The remaining architectures follow their respective original papers: DGCNN (Wang et al., 2019),
KPConv (Thomas et al., 2019), and Point Transformer (Zhao et al., 2021). Loss functions follow
Focal Loss (Lin et al., 2017), Tversky Loss (Salehi et al., 2017), and Lovász-Softmax
(Berman et al., 2018), as referenced in `src/models/losses.py`.

## Citation

```bibtex
@misc{damage_detection_3d,
  title  = {3D Damage Detection on Metallic Components},
  author = {[Author]},
  year   = {[Year]},
  note   = {[Publication / repository placeholder]}
}
```

## License

See [`LICENSE`](LICENSE).
