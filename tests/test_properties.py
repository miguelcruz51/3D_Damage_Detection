"""
Property-based tests for the dataset module.

Feature: repo-reorganization
Uses: hypothesis with numpy extras

Run with:
    pytest clean_repo/tests/test_properties.py -v
"""

import json
import os
import sys
import tempfile

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

# Make clean_repo/src importable from anywhere
_SRC = os.path.join(os.path.dirname(__file__), '..', 'src')
sys.path.insert(0, os.path.abspath(_SRC))

from data.preprocessing import pc_normalize
from data.dataset import DamageDataset


# ---------------------------------------------------------------------------
# Property 2: pc_normalize produces unit sphere
# Feature: repo-reorganization, Property 2: pc_normalize produces unit sphere
# — centroid ≈ 0, max_radius ≈ 1.0
# Validates: Requirements 3.4
# ---------------------------------------------------------------------------

@given(
    pc=arrays(
        dtype=np.float32,
        shape=st.tuples(
            st.integers(min_value=2, max_value=4096),  # N points
            st.just(3),                                  # 3D
        ),
        elements=st.floats(min_value=-1000.0, max_value=1000.0,
                           allow_nan=False, allow_infinity=False),
    )
)
@settings(max_examples=100)
def test_pc_normalize_unit_sphere(pc: np.ndarray) -> None:
    """
    # Feature: repo-reorganization, Property 2: pc_normalize produces unit sphere — centroid ≈ 0, max_radius ≈ 1.0
    # Validates: Requirements 3.4

    For any point cloud of shape (N, 3) with N >= 2 and finite float values,
    after applying pc_normalize:
      - the centroid should be the zero vector (within floating-point tolerance)
      - the maximum distance from the origin should be approximately 1.0
    """
    # Skip degenerate clouds whose spread is tiny relative to their magnitude.
    # pc_normalize divides by the (centered) max radius m; when m is dominated
    # by float32 rounding noise (e.g. all points identical, or a very small
    # spread on large-magnitude coordinates), dividing by (m + 1e-6) amplifies
    # that noise and the "unit sphere" property is undefined. Guard using the
    # same float32 computation pc_normalize performs, scaled by magnitude.
    pc32 = pc.astype(np.float32)
    centered = pc32 - pc32.mean(axis=0)
    m = float(np.max(np.sqrt(np.sum(centered ** 2, axis=1))))
    scale = float(np.max(np.abs(pc32))) + 1.0
    if m < 1e-3 * scale:
        return

    normalized = pc_normalize(pc.copy())

    centroid = np.mean(normalized, axis=0)
    max_radius = np.max(np.sqrt(np.sum(normalized ** 2, axis=1)))

    assert np.allclose(centroid, 0.0, atol=1e-4), (
        f"Centroid not near zero: {centroid}"
    )
    # pc_normalize divides by (m + 1e-6) for divide-by-zero safety, so the max
    # radius is m / (m + 1e-6): always slightly below 1.0, with the gap growing
    # for small m. Combined with float32 rounding, a 1e-4 tolerance is unrealistic
    # for tiny-magnitude clouds. 1e-3 keeps the property meaningful while
    # accounting for the deliberate epsilon and float32 precision.
    assert abs(max_radius - 1.0) < 1e-3, (
        f"Max radius not near 1.0: {max_radius}"
    )


# ---------------------------------------------------------------------------
# Property 3: DamageDataset split isolation
# Feature: repo-reorganization, Property 3: DamageDataset split isolation
# — for any split, all datapath entries belong exclusively to that split's JSON
# Validates: Requirements 3.2, 3.6
# ---------------------------------------------------------------------------

def _make_fake_dataset(
    tmpdir: str,
    n_train: int = 5,
    n_val: int = 2,
    n_test: int = 3,
) -> str:
    """Create a minimal fake dataset directory for testing."""
    components_dir = tmpdir

    # Write category mapping
    with open(os.path.join(components_dir, 'synsetoffset2category.txt'), 'w') as f:
        f.write("Component1\t001\n")

    # Create split directory
    split_dir = os.path.join(components_dir, 'train_test_split')
    os.makedirs(split_dir, exist_ok=True)

    # Create fake NPZ files and split lists
    comp_dir = os.path.join(components_dir, '001')
    os.makedirs(comp_dir, exist_ok=True)

    all_files = []
    for i in range(n_train + n_val + n_test):
        fname = f"001001{i:04d}0.npz"
        fpath = os.path.join(comp_dir, fname)
        if not os.path.exists(fpath):
            pts = np.random.randn(64, 3).astype(np.float32)
            nrm = np.random.randn(64, 3).astype(np.float32)
            lbl = np.zeros(64, dtype=np.int32)
            np.savez(fpath, points=pts, normals=nrm, labels=lbl)
        all_files.append(f"001/{fname}")

    train_files = all_files[:n_train]
    val_files = all_files[n_train:n_train + n_val]
    test_files = all_files[n_train + n_val:]

    with open(os.path.join(split_dir, 'shuffled_train_file_list.json'), 'w') as f:
        json.dump(train_files, f)
    with open(os.path.join(split_dir, 'shuffled_val_file_list.json'), 'w') as f:
        json.dump(val_files, f)
    with open(os.path.join(split_dir, 'shuffled_test_file_list.json'), 'w') as f:
        json.dump(test_files, f)

    return components_dir


@pytest.fixture(scope="module")
def fake_dataset_root(tmp_path_factory):
    """Shared fake dataset root for all split-isolation tests."""
    tmpdir = str(tmp_path_factory.mktemp("damaged_components"))
    return _make_fake_dataset(tmpdir, n_train=10, n_val=4, n_test=6)


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_split_isolation_no_overlap(fake_dataset_root: str, split: str) -> None:
    """
    # Feature: repo-reorganization, Property 3: DamageDataset split isolation
    # — for any split, all datapath entries should belong exclusively to that
    #   split's JSON file, with no overlap between train/val/test sets.
    # Validates: Requirements 3.2, 3.6

    For any split in {train, val, test}, all file paths in DamageDataset.datapath
    should:
      1. Be listed in the corresponding split JSON file.
      2. Not appear in the other two split JSON files.
    """
    split_dir = os.path.join(fake_dataset_root, 'train_test_split')

    with open(os.path.join(split_dir, 'shuffled_train_file_list.json')) as f:
        train_set = set(json.load(f))
    with open(os.path.join(split_dir, 'shuffled_val_file_list.json')) as f:
        val_set = set(json.load(f))
    with open(os.path.join(split_dir, 'shuffled_test_file_list.json')) as f:
        test_set = set(json.load(f))

    split_sets = {'train': train_set, 'val': val_set, 'test': test_set}
    other_splits = {s: sp for s, sp in split_sets.items() if s != split}

    ds = DamageDataset(root=fake_dataset_root, npoints=16, split=split)

    for rel_path in ds.datapath:
        # Must be in this split
        assert rel_path in split_sets[split], (
            f"Path '{rel_path}' not in {split} split JSON"
        )
        # Must not be in any other split
        for other_name, other_set in other_splits.items():
            assert rel_path not in other_set, (
                f"Path '{rel_path}' from {split} split also appears in {other_name} split"
            )


def test_split_isolation_all_splits_disjoint(fake_dataset_root: str) -> None:
    """
    # Feature: repo-reorganization, Property 3: DamageDataset split isolation
    # Validates: Requirements 3.2, 3.6

    For any dataset, the union of train/val/test datapaths should be
    disjoint (no file appears in two splits).
    """
    train_ds = DamageDataset(root=fake_dataset_root, npoints=16, split='train')
    val_ds = DamageDataset(root=fake_dataset_root, npoints=16, split='val')
    test_ds = DamageDataset(root=fake_dataset_root, npoints=16, split='test')

    train_set = set(train_ds.datapath)
    val_set = set(val_ds.datapath)
    test_set = set(test_ds.datapath)

    assert train_set.isdisjoint(val_set), "Train and val overlap"
    assert train_set.isdisjoint(test_set), "Train and test overlap"
    assert val_set.isdisjoint(test_set), "Val and test overlap"


def test_dataset_raises_on_missing_root() -> None:
    """DamageDataset must raise FileNotFoundError for a non-existent root."""
    with pytest.raises(FileNotFoundError):
        DamageDataset(root='/nonexistent/path/xyz', split='train')


# ---------------------------------------------------------------------------
# Property 5: All model forward passes produce correct output shapes [B, N, 2]
# Feature: repo-reorganization, Property 5: All model forward passes produce
# correct output shapes [B, N, 2]
# Validates: Requirements 4.4
# ---------------------------------------------------------------------------

import torch

from models import pointnet_dmg_seg
from models import pointnet2_dmg_seg_ssg
from models import dgcnn_dmg_seg
from models import kpconv_dmg_seg_v2
from models import point_transformer_dmg_seg_v3
from models import pointnet_transfer_dmg_seg
from models import pointnet2_transfer_dmg_seg


def _build_model(name: str, normal_channel: bool):
    """
    Instantiate a model by name and return (model, in_channels).

    Encapsulates the per-model constructor differences so the property test
    can treat every architecture uniformly.
    """
    if name == "pointnet":
        model = pointnet_dmg_seg.get_model(part_num=2, normal_channel=normal_channel)
        channels = 6 if normal_channel else 3
    elif name == "pointnet2_ssg":
        model = pointnet2_dmg_seg_ssg.get_model(normal_channel=normal_channel)
        channels = 6 if normal_channel else 3
    elif name == "dgcnn":
        model = dgcnn_dmg_seg.get_model(num_classes=2, k=20, normal_channel=normal_channel)
        channels = 6 if normal_channel else 3
    elif name == "kpconv":
        model = kpconv_dmg_seg_v2.get_model(num_classes=2, normal_channel=normal_channel)
        channels = 6 if normal_channel else 3
    elif name == "point_transformer":
        model = point_transformer_dmg_seg_v3.get_model(num_classes=2, normal_channel=normal_channel)
        channels = 6 if normal_channel else 3
    elif name == "pointnet_transfer":
        # Transfer PointNet always uses a 6-channel (XYZ+normals) encoder.
        model = pointnet_transfer_dmg_seg.get_model(part_num=2)
        channels = 6
    elif name == "pointnet2_transfer":
        # The PointNet2 transfer model is fine-tuned from an S3DIS backbone and
        # only supports XYZ+normals input. Its sa1 layer expects grouped
        # features of width (input_channels + 3) = 9, i.e. a 6-channel input.
        model = pointnet2_transfer_dmg_seg.get_model(num_classes=2, normal_channel=True)
        channels = 6
    else:
        raise ValueError(f"Unknown model name: {name}")

    return model.eval(), channels


_MODEL_NAMES = [
    "pointnet",
    "pointnet2_ssg",
    "dgcnn",
    "kpconv",
    "point_transformer",
    "pointnet_transfer",
    "pointnet2_transfer",
]


@settings(max_examples=100, deadline=None)
@given(
    name=st.sampled_from(_MODEL_NAMES),
    normal_channel=st.booleans(),
    B=st.integers(min_value=1, max_value=2),
    N=st.integers(min_value=256, max_value=1024),
)
def test_model_forward_output_shapes(name: str, normal_channel: bool,
                                     B: int, N: int) -> None:
    """
    # Feature: repo-reorganization, Property 5: All model forward passes produce correct output shapes [B, N, 2]
    # Validates: Requirements 4.4

    For any model in src/models/ instantiated via get_model(), calling
    forward(xyz) with a random input of shape [B, C, N] (C=3, 6, or 9
    depending on the architecture and normal_channel) should return a tuple
    whose first element has shape [B, N, 2].
    """
    torch.manual_seed(0)
    model, channels = _build_model(name, normal_channel)

    xyz = torch.randn(B, channels, N)

    with torch.no_grad():
        out = model(xyz)

    assert isinstance(out, tuple), f"{name}: forward should return a tuple"
    log_probs = out[0]
    assert log_probs.shape == (B, N, 2), (
        f"{name} (normal_channel={normal_channel}, B={B}, N={N}): "
        f"expected output shape {(B, N, 2)}, got {tuple(log_probs.shape)}"
    )


# ---------------------------------------------------------------------------
# Property 8: All model files in src/models/ have non-empty module-level docstrings
# Feature: repo-reorganization, Property 8: All model files in src/models/ have
# non-empty module-level docstrings
# Validates: Requirements 4.2
# ---------------------------------------------------------------------------

import importlib
import glob

_MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'models'))

# Files excluded per the design: package init and low-level utility modules
# that are not model-architecture files.
_DOCSTRING_EXCLUDED = {"__init__.py", "pointnet2_utils.py"}


def _model_module_files() -> list[str]:
    """Enumerate candidate model architecture .py files in src/models/."""
    files = []
    for path in sorted(glob.glob(os.path.join(_MODELS_DIR, "*.py"))):
        base = os.path.basename(path)
        if base in _DOCSTRING_EXCLUDED:
            continue
        files.append(base)
    return files


@given(module_file=st.sampled_from(_model_module_files()))
@settings(max_examples=100)
def test_model_module_has_docstring(module_file: str) -> None:
    """
    # Feature: repo-reorganization, Property 8: All model files in src/models/ have non-empty module-level docstrings
    # Validates: Requirements 4.2

    For any Python file in clean_repo/src/models/ (excluding __init__.py and
    pointnet2_utils.py), the module should have a non-empty module-level
    docstring accessible via module.__doc__.
    """
    module_name = "models." + os.path.splitext(module_file)[0]
    module = importlib.import_module(module_name)

    doc = module.__doc__
    assert doc is not None, f"{module_file}: missing module-level docstring"
    assert doc.strip() != "", f"{module_file}: module-level docstring is empty"


# ---------------------------------------------------------------------------
# Property 6: Stochastic voting — aggregated probability equals mean of
# sub-sample probabilities
# Feature: repo-reorganization, Property 6: Stochastic voting produces mean of
# sub-sample probabilities
# Validates: Requirements 7.4
# ---------------------------------------------------------------------------

from evaluation.inference import aggregate_votes


@st.composite
def _vote_maps(draw):
    """Generate a (prob_acc, n_points) pair with random per-index vote lists."""
    n_points = draw(st.integers(min_value=1, max_value=50))
    # Choose a subset of point indices that receive votes
    voted_indices = draw(
        st.lists(
            st.integers(min_value=0, max_value=n_points - 1),
            min_size=0, max_size=n_points, unique=True,
        )
    )
    prob_acc = {}
    for idx in voted_indices:
        votes = draw(
            st.lists(
                st.floats(min_value=0.0, max_value=1.0,
                          allow_nan=False, allow_infinity=False),
                min_size=1, max_size=8,
            )
        )
        prob_acc[idx] = votes
    return prob_acc, n_points


@given(data=_vote_maps())
@settings(max_examples=100)
def test_stochastic_voting_is_mean(data) -> None:
    """
    # Feature: repo-reorganization, Property 6: Stochastic voting produces mean of sub-sample probabilities
    # Validates: Requirements 7.4

    For any set of probability votes accumulated per point index, the
    aggregated probability for each point should equal the mean of all
    probabilities assigned to that point; unvoted points should be 0.0.
    """
    prob_acc, n_points = data
    result = aggregate_votes(prob_acc, n_points)

    assert result.shape == (n_points,)
    for idx in range(n_points):
        if idx in prob_acc:
            expected = float(np.mean(prob_acc[idx]))
            assert abs(result[idx] - expected) < 1e-9, (
                f"Point {idx}: expected mean {expected}, got {result[idx]}"
            )
        else:
            assert result[idx] == 0.0, f"Unvoted point {idx} should be 0.0"


# ---------------------------------------------------------------------------
# Property 9: metrics.py produces values in [0, 1] range for any valid binary
# label inputs
# Feature: repo-reorganization, Property 9: metrics.py produces values in
# [0, 1] range
# Validates: Requirements 6.2
# ---------------------------------------------------------------------------

from evaluation.metrics import calculate_metrics


@given(
    n=st.integers(min_value=1, max_value=2000),
    true_ratio=st.floats(min_value=0.0, max_value=1.0),
    pred_ratio=st.floats(min_value=0.0, max_value=1.0),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=100)
def test_metrics_in_unit_range(n: int, true_ratio: float,
                               pred_ratio: float, seed: int) -> None:
    """
    # Feature: repo-reorganization, Property 9: metrics.py produces values in [0, 1] range
    # Validates: Requirements 6.2

    For any valid pair of binary label arrays (pred and true) with varying
    class ratios, all bounded metrics returned by calculate_metrics
    (precision, recall, f1, iou_intact, iou_damaged, miou, dice, dice_damaged,
    pr_auc) should be in the range [0.0, 1.0].
    """
    rng = np.random.default_rng(seed)
    true_labels = (rng.random(n) < true_ratio).astype(np.int32)
    pred_labels = (rng.random(n) < pred_ratio).astype(np.int32)
    pred_scores = rng.random(n)

    _, _, _, _, _, results = calculate_metrics(pred_labels, true_labels, pred_scores)

    bounded_keys = [
        "precision", "recall", "f1", "iou_intact", "iou_damaged",
        "miou", "dice", "dice_damaged", "pr_auc",
    ]
    # sklearn's average_precision_score can return values marginally outside
    # [0, 1] (e.g. 1.0000000000000002) due to float64 rounding. Allow a small
    # tolerance so the property tests the meaningful bound, not float artifacts.
    tol = 1e-9
    for key in bounded_keys:
        value = results[key]
        if value is None:
            continue
        assert -tol <= value <= 1.0 + tol, f"Metric '{key}' = {value} outside [0, 1]"


# ---------------------------------------------------------------------------
# Property 1: No absolute local paths in any source file under clean_repo/
# Feature: repo-reorganization, Property 1: No absolute local paths in any
# source file under clean_repo/
# Validates: Requirements 2.4, 7.6, 9.1
# ---------------------------------------------------------------------------

_CLEAN_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Machine-specific absolute path fragments that must never appear in source.
_FORBIDDEN_PATH_FRAGMENTS = [
    "/home/miguelcruz",
    "/home/miguel",
]


def _clean_repo_py_files() -> list[str]:
    """Enumerate .py source files under clean_repo/ (excluding caches and tests).

    The ``tests/`` directory is excluded because the test harness itself
    legitimately references forbidden path fragments as string literals; the
    requirements (2.4, 7.6, 9.1) concern shippable source files, not tests.
    """
    files = []
    for dirpath, dirnames, filenames in os.walk(_CLEAN_REPO_ROOT):
        # Skip caches, virtual environments, and the tests directory itself
        dirnames[:] = [
            d for d in dirnames
            if d not in ("__pycache__", ".pytest_cache", ".hypothesis",
                         "venv", ".git", "tests")
        ]
        for fname in filenames:
            if fname.endswith(".py"):
                files.append(os.path.join(dirpath, fname))
    return sorted(files)


@given(py_file=st.sampled_from(_clean_repo_py_files()))
@settings(max_examples=100)
def test_no_absolute_local_paths(py_file: str) -> None:
    """
    # Feature: repo-reorganization, Property 1: No absolute local paths in any source file under clean_repo/
    # Validates: Requirements 2.4, 7.6, 9.1

    For any Python source file placed under clean_repo/, the file's content
    should not contain any machine-specific absolute path fragment such as
    /home/miguelcruz.
    """
    with open(py_file, encoding="utf-8", errors="ignore") as fh:
        content = fh.read()

    for fragment in _FORBIDDEN_PATH_FRAGMENTS:
        assert fragment not in content, (
            f"{os.path.relpath(py_file, _CLEAN_REPO_ROOT)} contains "
            f"forbidden absolute path fragment '{fragment}'"
        )


# ---------------------------------------------------------------------------
# Property 4: All src/ subdirectories have __init__.py
# Feature: repo-reorganization, Property 4: All src/ subdirectories have
# __init__.py
# Validates: Requirements 2.3
# ---------------------------------------------------------------------------

_SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))


def _src_subdirectories() -> list[str]:
    """Enumerate every subdirectory under clean_repo/src/ (excluding caches)."""
    dirs = []
    for dirpath, dirnames, _ in os.walk(_SRC_ROOT):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        if os.path.abspath(dirpath) == os.path.abspath(_SRC_ROOT):
            # src/ itself is covered separately; include package subdirs only
            continue
        dirs.append(dirpath)
    return sorted(dirs)


@given(subdir=st.sampled_from(_src_subdirectories()))
@settings(max_examples=100)
def test_src_subdirs_have_init(subdir: str) -> None:
    """
    # Feature: repo-reorganization, Property 4: All src/ subdirectories have __init__.py
    # Validates: Requirements 2.3

    For any subdirectory under clean_repo/src/, an __init__.py file should be
    present, making the directory a valid importable Python package.
    """
    init_path = os.path.join(subdir, "__init__.py")
    assert os.path.isfile(init_path), (
        f"Missing __init__.py in {os.path.relpath(subdir, _SRC_ROOT)}"
    )


# ---------------------------------------------------------------------------
# Property 7: No Python file in clean_repo/ consists entirely of commented-out
# code
# Feature: repo-reorganization, Property 7: No entirely-dead-code Python files
# in clean_repo
# Validates: Requirements 10.5
# ---------------------------------------------------------------------------

def _is_executable_line(line: str) -> bool:
    """Return True if a line is neither blank nor a comment."""
    stripped = line.strip()
    if not stripped:
        return False
    if stripped.startswith("#"):
        return False
    return True


@given(py_file=st.sampled_from(_clean_repo_py_files()))
@settings(max_examples=100)
def test_no_dead_code_files(py_file: str) -> None:
    """
    # Feature: repo-reorganization, Property 7: No entirely-dead-code Python files in clean_repo
    # Validates: Requirements 10.5

    For any Python file in clean_repo/ that contains commented-out code, the
    count of executable (non-comment, non-blank) lines should be greater than
    0, ensuring no file consists entirely of commented-out code.

    Empty or blank-only files (e.g. empty package-marker __init__.py) are
    not "commented-out dead code" and are outside this property's domain.
    """
    with open(py_file, encoding="utf-8", errors="ignore") as fh:
        lines = fh.readlines()

    executable = sum(1 for line in lines if _is_executable_line(line))
    comment_lines = sum(1 for line in lines if line.strip().startswith("#"))

    # Only files that actually contain commented-out code are in scope; a file
    # with no comment lines and no executable lines is simply an empty marker.
    if comment_lines == 0:
        return

    assert executable > 0, (
        f"{os.path.relpath(py_file, _CLEAN_REPO_ROOT)} consists entirely of "
        "commented-out code (no executable lines)"
    )


# ---------------------------------------------------------------------------
# Property 10: pc_normalize from preprocessing.py produces identical output to
# the inline normalization in dataset.py
# Feature: repo-reorganization, Property 10: Normalization consistency between
# training and inference
# Validates: Requirements 7.3
# ---------------------------------------------------------------------------

from data.dataset import _pc_normalize as dataset_pc_normalize


@given(
    pc=arrays(
        dtype=np.float32,
        shape=st.tuples(
            st.integers(min_value=1, max_value=4096),  # N points
            st.just(3),                                  # 3D
        ),
        elements=st.floats(min_value=-1000.0, max_value=1000.0,
                           allow_nan=False, allow_infinity=False),
    )
)
@settings(max_examples=100)
def test_normalization_consistency(pc: np.ndarray) -> None:
    """
    # Feature: repo-reorganization, Property 10: Normalization consistency between training and inference
    # Validates: Requirements 7.3

    For any point cloud array, pc_normalize from src/data/preprocessing.py
    (used during inference) should produce output identical to the inline
    _pc_normalize used inside DamageDataset.__getitem__ (used during training).
    """
    from_preprocessing = pc_normalize(pc.copy())
    from_dataset = dataset_pc_normalize(pc.copy())

    assert from_preprocessing.shape == from_dataset.shape
    assert np.allclose(from_preprocessing, from_dataset, atol=1e-6, equal_nan=True), (
        "preprocessing.pc_normalize and dataset._pc_normalize disagree"
    )
