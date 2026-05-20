"""Utilities for discovering and lazily reading local active matter datasets."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import h5py
import numpy as np

try:
    import xarray as xr
except Exception:  # pragma: no cover - optional dependency at runtime
    xr = None


DATA_EXTENSIONS = (".hdf5", ".h5", ".nc", ".nc4", ".cdf")
PARAM_RE = re.compile(
    r"zeta_(?P<zeta>-?\d+(?:\.\d+)?)_alpha_(?P<alpha>-?\d+(?:\.\d+)?)"
)


@dataclass
class DataFileRecord:
    """Flat file inventory record."""

    path: str
    relpath: str
    split: Optional[str]
    suffix: str
    size_bytes: int
    alpha_from_name: Optional[float] = None
    zeta_from_name: Optional[float] = None
    format: Optional[str] = None
    layout: Optional[str] = None
    n_samples: Optional[int] = None


@dataclass
class FieldSpec:
    """Description of one logical field in a container."""

    group: str
    dataset_path: str
    field_name: str
    kind: str
    shape: Tuple[int, ...]
    dtype: str
    component_shape: Tuple[int, ...] = field(default_factory=tuple)
    channel_names: List[str] = field(default_factory=list)


@dataclass
class FileMetadata:
    """Metadata extracted from a single HDF5 or NetCDF file."""

    path: str
    format: str
    layout: str
    split: Optional[str]
    size_bytes: int
    attrs: Dict[str, Any]
    dimensions: Dict[str, Any]
    fields: List[FieldSpec]
    tree_rows: List[Dict[str, Any]]
    label_info: Dict[str, Any]
    sample_axis: Optional[str]
    n_samples: Optional[int]
    detected_channel_names: List[str]
    uncertainties: List[str] = field(default_factory=list)


def ensure_dir(path: Path) -> Path:
    """Create a directory and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def json_ready(value: Any) -> Any:
    """Convert HDF5/xarray metadata into JSON-safe objects."""
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    return value


def discover_data_files(data_root: Path, max_files: Optional[int] = None) -> List[Path]:
    """Recursively discover supported container files."""
    files: List[Path] = []
    for suffix in DATA_EXTENSIONS:
        files.extend(data_root.rglob(f"*{suffix}"))
    files = sorted(set(files))
    if max_files is not None:
        files = files[:max_files]
    return files


def balanced_file_subset(files: Sequence[Path], data_root: Path, limit: int) -> List[Path]:
    """Round-robin sample files across inferred splits for representative quick scans."""
    if limit >= len(files):
        return list(files)
    buckets: Dict[str, List[Path]] = {}
    for path in files:
        split = infer_split_from_path(path, data_root) or "unspecified"
        buckets.setdefault(split, []).append(path)
    keys = sorted(buckets)
    selected: List[Path] = []
    cursor = 0
    while len(selected) < limit and any(buckets.values()):
        key = keys[cursor % len(keys)]
        if buckets[key]:
            selected.append(buckets[key].pop(0))
        cursor += 1
    return selected[:limit]


def infer_split_from_path(path: Path, data_root: Path) -> Optional[str]:
    """Infer dataset split from directory names."""
    rel = path.relative_to(data_root)
    for part in rel.parts[:-1]:
        if part.lower() in {"train", "valid", "val", "test"}:
            return "valid" if part.lower() == "val" else part.lower()
    return None


def extract_params_from_name(path: Path) -> Dict[str, Optional[float]]:
    """Extract alpha/zeta if encoded in the filename."""
    match = PARAM_RE.search(path.name)
    if not match:
        return {"alpha": None, "zeta": None}
    return {
        "alpha": float(match.group("alpha")),
        "zeta": float(match.group("zeta")),
    }


def file_inventory_record(path: Path, data_root: Path) -> DataFileRecord:
    """Create a file inventory record with name-derived metadata."""
    params = extract_params_from_name(path)
    return DataFileRecord(
        path=str(path.resolve()),
        relpath=str(path.resolve().relative_to(data_root.resolve())),
        split=infer_split_from_path(path, data_root),
        suffix=path.suffix.lower(),
        size_bytes=path.stat().st_size,
        alpha_from_name=params["alpha"],
        zeta_from_name=params["zeta"],
    )


def _decode_listish(values: Any) -> List[str]:
    out: List[str] = []
    if values is None:
        return out
    if isinstance(values, (str, bytes)):
        values = [values]
    for value in values:
        if isinstance(value, bytes):
            out.append(value.decode("utf-8", errors="replace"))
        else:
            out.append(str(value))
    return out


def _tensor_component_names(field_name: str, component_shape: Sequence[int]) -> List[str]:
    if not component_shape:
        return [field_name]
    if tuple(component_shape) == (2,):
        return [f"{field_name}_x", f"{field_name}_y"]
    if tuple(component_shape) == (3,):
        return [f"{field_name}_{axis}" for axis in ("x", "y", "z")]
    if tuple(component_shape) == (2, 2):
        return [f"{field_name}_{i}{j}" for i in range(2) for j in range(2)]
    n_components = int(np.prod(component_shape))
    return [f"{field_name}_{idx}" for idx in range(n_components)]


def _hdf5_tree_rows(handle: h5py.File) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    def visitor(name: str, obj: Any) -> None:
        row = {
            "path": "/" if name == "" else name,
            "kind": "group" if isinstance(obj, h5py.Group) else "dataset",
            "shape": list(obj.shape) if isinstance(obj, h5py.Dataset) else None,
            "dtype": str(obj.dtype) if isinstance(obj, h5py.Dataset) else None,
            "attrs": {str(k): json_ready(v) for k, v in obj.attrs.items()},
        }
        rows.append(row)

    handle.visititems(visitor)
    return rows


def _infer_hdf5_layout(handle: h5py.File, fields: List[FieldSpec]) -> Tuple[str, Optional[str], Optional[int], List[str]]:
    uncertainties: List[str] = []
    if {"t0_fields", "t1_fields", "t2_fields"}.intersection(handle.keys()):
        n_samples = int(handle.attrs.get("n_trajectories", 0) or 0)
        if n_samples == 0:
            for spec in fields:
                if len(spec.shape) >= 1:
                    n_samples = int(spec.shape[0])
                    uncertainties.append(
                        "n_trajectories attribute missing; inferred sample axis from field shape."
                    )
                    break
        return "raw_trajectory_fields", "trajectory", n_samples or None, uncertainties

    candidate_datasets = [spec for spec in fields if len(spec.shape) >= 4]
    if candidate_datasets:
        first = candidate_datasets[0]
        sample_dim = int(first.shape[0])
        time_dim = int(first.shape[1]) if len(first.shape) >= 2 else None
        spatial_shape = first.shape[-2:]
        if time_dim in {8, 12, 16, 24, 32} and all(v in {64, 128, 224, 256} for v in spatial_shape):
            return "processed_windows", "window", sample_dim, uncertainties
        return "dense_tensor_samples", "sample", sample_dim, uncertainties

    return "unknown", None, None, ["Could not confidently infer layout from HDF5 tree."]


def inspect_hdf5_file(path: Path, data_root: Optional[Path] = None) -> FileMetadata:
    """Extract structure metadata from a HDF5 file."""
    with h5py.File(path, "r") as handle:
        attrs = {str(k): json_ready(v) for k, v in handle.attrs.items()}
        tree_rows = _hdf5_tree_rows(handle)
        dimensions: Dict[str, Any] = {}
        if "dimensions" in handle:
            for key, ds in handle["dimensions"].items():
                if isinstance(ds, h5py.Dataset):
                    dimensions[key] = {"shape": list(ds.shape), "dtype": str(ds.dtype)}

        fields: List[FieldSpec] = []
        detected_channel_names: List[str] = []
        for group_name, kind in (("t0_fields", "scalar"), ("t1_fields", "vector"), ("t2_fields", "tensor")):
            if group_name not in handle:
                continue
            group = handle[group_name]
            field_names = _decode_listish(group.attrs.get("field_names"))
            dataset_names = list(group.keys())
            if not field_names:
                field_names = dataset_names
            for idx, dataset_name in enumerate(dataset_names):
                field_name = field_names[idx] if idx < len(field_names) else dataset_name
                dataset = group[dataset_name]
                base_dims = dataset.shape
                component_shape: Tuple[int, ...] = ()
                if kind == "vector" and dataset.ndim >= 1:
                    component_shape = (dataset.shape[-1],)
                elif kind == "tensor" and dataset.ndim >= 2:
                    component_shape = tuple(dataset.shape[-2:])
                channel_names = _tensor_component_names(field_name, component_shape)
                detected_channel_names.extend(channel_names)
                fields.append(
                    FieldSpec(
                        group=group_name,
                        dataset_path=f"{group_name}/{dataset_name}",
                        field_name=field_name,
                        kind=kind,
                        shape=tuple(int(v) for v in base_dims),
                        dtype=str(dataset.dtype),
                        component_shape=component_shape,
                        channel_names=channel_names,
                    )
                )

        if not fields:
            for key, obj in handle.items():
                if isinstance(obj, h5py.Dataset):
                    fields.append(
                        FieldSpec(
                            group="/",
                            dataset_path=key,
                            field_name=key,
                            kind="unknown",
                            shape=tuple(int(v) for v in obj.shape),
                            dtype=str(obj.dtype),
                            component_shape=(),
                            channel_names=[key],
                        )
                    )
                    detected_channel_names.append(key)

        layout, sample_axis, n_samples, uncertainties = _infer_hdf5_layout(handle, fields)
        label_info: Dict[str, Any] = {}
        name_params = extract_params_from_name(path)
        if "scalars" in handle:
            for key, ds in handle["scalars"].items():
                if isinstance(ds, h5py.Dataset) and ds.shape == ():
                    label_info[key] = json_ready(ds[()])
        for key in ("alpha", "zeta"):
            if key in attrs:
                label_info[key] = attrs[key]
            elif key in name_params and name_params[key] is not None:
                label_info[key] = name_params[key]
        split = infer_split_from_path(path, data_root or path.parent)

    return FileMetadata(
        path=str(path.resolve()),
        format="hdf5",
        layout=layout,
        split=split,
        size_bytes=path.stat().st_size,
        attrs=attrs,
        dimensions=dimensions,
        fields=fields,
        tree_rows=tree_rows,
        label_info=label_info,
        sample_axis=sample_axis,
        n_samples=n_samples,
        detected_channel_names=detected_channel_names,
        uncertainties=uncertainties,
    )


def inspect_netcdf_file(path: Path, data_root: Optional[Path] = None) -> FileMetadata:
    """Extract structure metadata from a NetCDF file."""
    if xr is None:
        raise RuntimeError("xarray is not available; cannot inspect NetCDF files.")
    with xr.open_dataset(path) as ds:
        tree_rows = []
        detected_channel_names = []
        fields: List[FieldSpec] = []
        for name, var in ds.data_vars.items():
            shape = tuple(int(v) for v in var.shape)
            fields.append(
                FieldSpec(
                    group="/",
                    dataset_path=name,
                    field_name=name,
                    kind="unknown",
                    shape=shape,
                    dtype=str(var.dtype),
                    component_shape=(),
                    channel_names=[name],
                )
            )
            detected_channel_names.append(name)
            tree_rows.append(
                {
                    "path": name,
                    "kind": "variable",
                    "shape": list(shape),
                    "dtype": str(var.dtype),
                    "attrs": {str(k): json_ready(v) for k, v in var.attrs.items()},
                }
            )
        dimensions = {str(k): int(v) for k, v in ds.dims.items()}
        attrs = {str(k): json_ready(v) for k, v in ds.attrs.items()}
        layout = "unknown"
        sample_axis = None
        n_samples = None
        uncertainties = ["NetCDF support is generic; layout inference may be incomplete."]
        split = infer_split_from_path(path, data_root or path.parent)
        label_info = {
            key: attrs.get(key)
            for key in ("alpha", "zeta")
            if key in attrs
        }
    return FileMetadata(
        path=str(path.resolve()),
        format="netcdf",
        layout=layout,
        split=split,
        size_bytes=path.stat().st_size,
        attrs=attrs,
        dimensions=dimensions,
        fields=fields,
        tree_rows=tree_rows,
        label_info=label_info,
        sample_axis=sample_axis,
        n_samples=n_samples,
        detected_channel_names=detected_channel_names,
        uncertainties=uncertainties,
    )


def inspect_data_file(path: Path, data_root: Optional[Path] = None) -> FileMetadata:
    """Dispatch to the correct inspector based on file suffix."""
    suffix = path.suffix.lower()
    if suffix in {".h5", ".hdf5"}:
        return inspect_hdf5_file(path, data_root=data_root)
    if suffix in {".nc", ".nc4", ".cdf"}:
        return inspect_netcdf_file(path, data_root=data_root)
    raise ValueError(f"Unsupported data file: {path}")


def metadata_to_dict(meta: FileMetadata) -> Dict[str, Any]:
    """Convert metadata dataclass to a JSON-safe dictionary."""
    return {
        "path": meta.path,
        "format": meta.format,
        "layout": meta.layout,
        "split": meta.split,
        "size_bytes": meta.size_bytes,
        "attrs": json_ready(meta.attrs),
        "dimensions": json_ready(meta.dimensions),
        "fields": [
            {
                "group": spec.group,
                "dataset_path": spec.dataset_path,
                "field_name": spec.field_name,
                "kind": spec.kind,
                "shape": list(spec.shape),
                "dtype": spec.dtype,
                "component_shape": list(spec.component_shape),
                "channel_names": list(spec.channel_names),
            }
            for spec in meta.fields
        ],
        "tree_rows": json_ready(meta.tree_rows),
        "label_info": json_ready(meta.label_info),
        "sample_axis": meta.sample_axis,
        "n_samples": meta.n_samples,
        "detected_channel_names": list(meta.detected_channel_names),
        "uncertainties": list(meta.uncertainties),
    }


def parse_channel_names(meta: FileMetadata) -> List[str]:
    """Return inferred channel/component names."""
    if meta.detected_channel_names:
        return meta.detected_channel_names
    names: List[str] = []
    for spec in meta.fields:
        names.extend(spec.channel_names or [spec.field_name])
    return names


def read_raw_hdf5_sample(path: Path, sample_idx: int, sample_stride: int = 1) -> Dict[str, Any]:
    """Read one trajectory sample from a Well-style raw HDF5 file."""
    with h5py.File(path, "r") as handle:
        channels: List[np.ndarray] = []
        channel_names: List[str] = []
        field_sources: List[str] = []
        for group_name, kind in (("t0_fields", "scalar"), ("t1_fields", "vector"), ("t2_fields", "tensor")):
            if group_name not in handle:
                continue
            group = handle[group_name]
            field_names = _decode_listish(group.attrs.get("field_names"))
            dataset_names = list(group.keys())
            if not field_names:
                field_names = dataset_names
            for idx, dataset_name in enumerate(dataset_names):
                field_name = field_names[idx] if idx < len(field_names) else dataset_name
                dataset = group[dataset_name]
                selection = dataset[sample_idx, ::sample_stride]
                if kind == "scalar":
                    channels.append(selection[..., np.newaxis])
                    channel_names.append(field_name)
                    field_sources.append(dataset_name)
                else:
                    if kind == "vector":
                        component_names = _tensor_component_names(field_name, (selection.shape[-1],))
                        flat = selection.reshape(selection.shape[:-1] + (-1,))
                    else:
                        component_names = _tensor_component_names(field_name, selection.shape[-2:])
                        flat = selection.reshape(selection.shape[:-2] + (-1,))
                    channels.append(flat)
                    channel_names.extend(component_names)
                    field_sources.extend([dataset_name] * flat.shape[-1])
        if not channels:
            raise ValueError(f"Unsupported raw HDF5 layout in {path}")
        array = np.concatenate(channels, axis=-1).astype(np.float32, copy=False)
        dimensions = {
            "time": handle["dimensions/time"][::sample_stride] if "dimensions" in handle and "time" in handle["dimensions"] else None,
            "x": handle["dimensions/x"][:] if "dimensions" in handle and "x" in handle["dimensions"] else None,
            "y": handle["dimensions/y"][:] if "dimensions" in handle and "y" in handle["dimensions"] else None,
        }
        labels = {
            "alpha": float(handle.attrs["alpha"]) if "alpha" in handle.attrs else None,
            "zeta": float(handle.attrs["zeta"]) if "zeta" in handle.attrs else None,
            "L": float(handle.attrs["L"]) if "L" in handle.attrs else None,
        }
        return {
            "array": array,
            "channel_names": channel_names,
            "field_sources": field_sources,
            "dimensions": {k: json_ready(v) for k, v in dimensions.items()},
            "labels": labels,
            "layout": "raw_trajectory_fields",
        }


def iter_file_samples(
    path: Path,
    metadata: FileMetadata,
    max_samples: Optional[int] = None,
    sample_stride: int = 1,
) -> Iterator[Dict[str, Any]]:
    """Iterate lazily over logical samples within one file."""
    limit = metadata.n_samples if max_samples is None else min(metadata.n_samples or 0, max_samples)
    if metadata.format == "hdf5" and metadata.layout == "raw_trajectory_fields":
        for sample_idx in range(limit or 0):
            sample = read_raw_hdf5_sample(path, sample_idx=sample_idx, sample_stride=sample_stride)
            sample.update(
                {
                    "sample_index": sample_idx,
                    "file_path": str(path.resolve()),
                    "split": metadata.split,
                }
            )
            yield sample
        return

    if metadata.format == "netcdf" and xr is not None:
        with xr.open_dataset(path) as ds:
            sample_dim = next(iter(ds.dims), None)
            if sample_dim is None:
                return
            n = int(ds.dims[sample_dim])
            stop = n if max_samples is None else min(n, max_samples)
            for sample_idx in range(stop):
                arrs = []
                channel_names = []
                for name, var in ds.data_vars.items():
                    data = var.isel({sample_dim: sample_idx}).values
                    if data.ndim == 3:
                        arrs.append(data[..., np.newaxis])
                        channel_names.append(name)
                    else:
                        flat = data.reshape(data.shape[:-1] + (-1,))
                        arrs.append(flat)
                        channel_names.extend([f"{name}_{i}" for i in range(flat.shape[-1])])
                if not arrs:
                    continue
                yield {
                    "array": np.concatenate(arrs, axis=-1).astype(np.float32, copy=False),
                    "channel_names": channel_names,
                    "field_sources": channel_names,
                    "dimensions": {str(k): int(v) for k, v in ds.dims.items()},
                    "labels": {
                        "alpha": ds.attrs.get("alpha"),
                        "zeta": ds.attrs.get("zeta"),
                    },
                    "layout": metadata.layout,
                    "sample_index": sample_idx,
                    "file_path": str(path.resolve()),
                    "split": metadata.split,
                }
        return

    raise ValueError(f"Unsupported sample iteration for {path} with layout {metadata.layout}")


def iter_dataset_samples(
    files: Sequence[Path],
    metadata_map: Dict[str, FileMetadata],
    max_files: Optional[int] = None,
    max_samples_per_file: Optional[int] = None,
    sample_stride: int = 1,
) -> Iterator[Dict[str, Any]]:
    """Iterate over samples from multiple files without loading everything into memory."""
    selected = files if max_files is None else files[:max_files]
    for path in selected:
        meta = metadata_map[str(path.resolve())]
        yield from iter_file_samples(
            path,
            metadata=meta,
            max_samples=max_samples_per_file,
            sample_stride=sample_stride,
        )


def summarize_layouts(metadata_list: Sequence[FileMetadata]) -> Dict[str, Any]:
    """Create high-level counts about detected layouts and labels."""
    split_counts: Dict[str, int] = {}
    layout_counts: Dict[str, int] = {}
    format_counts: Dict[str, int] = {}
    alpha_values: set = set()
    zeta_values: set = set()
    combos: set = set()
    total_samples = 0

    for meta in metadata_list:
        split = meta.split or "unspecified"
        split_counts[split] = split_counts.get(split, 0) + 1
        layout_counts[meta.layout] = layout_counts.get(meta.layout, 0) + 1
        format_counts[meta.format] = format_counts.get(meta.format, 0) + 1
        total_samples += int(meta.n_samples or 0)
        alpha = meta.label_info.get("alpha")
        zeta = meta.label_info.get("zeta")
        if alpha is not None:
            alpha_values.add(float(alpha))
        if zeta is not None:
            zeta_values.add(float(zeta))
        if alpha is not None and zeta is not None:
            combos.add((float(alpha), float(zeta)))

    return {
        "num_files": len(metadata_list),
        "split_file_counts": split_counts,
        "layout_counts": layout_counts,
        "format_counts": format_counts,
        "unique_alpha_values": sorted(alpha_values),
        "unique_zeta_values": sorted(zeta_values),
        "num_alpha_values": len(alpha_values),
        "num_zeta_values": len(zeta_values),
        "num_alpha_zeta_combinations": len(combos),
        "total_declared_samples": total_samples,
    }


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    """Write indented JSON."""
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True))


def size_to_gb(size_bytes: int) -> float:
    """Convert bytes to GiB for display."""
    return size_bytes / math.pow(1024.0, 3)
