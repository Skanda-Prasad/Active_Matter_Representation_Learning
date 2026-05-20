"""Inspect the local active matter dataset layout and save an audit."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd
from tqdm import tqdm

from utils_io import (
    balanced_file_subset,
    discover_data_files,
    ensure_dir,
    file_inventory_record,
    inspect_data_file,
    metadata_to_dict,
    save_json,
    size_to_gb,
    summarize_layouts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, required=True, help="Dataset root directory")
    parser.add_argument("--out_dir", type=Path, default=Path("forensics_outputs"))
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--quick_mode", action="store_true", help="Inspect fewer files for a fast first pass")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    audit_dir = ensure_dir(out_dir / "audit")

    limit = args.max_files
    if args.quick_mode and limit is None:
        limit = 12

    files = discover_data_files(args.data_root)
    if limit is not None:
        files = balanced_file_subset(files, args.data_root, limit)
    if not files:
        raise SystemExit(f"No HDF5/NetCDF files found under {args.data_root}")

    inventory_rows = []
    metadata_dicts = []
    metadata_objects = []

    for path in tqdm(files, desc="Inspecting files"):
        record = file_inventory_record(path, args.data_root)
        meta = inspect_data_file(path, data_root=args.data_root)
        record.format = meta.format
        record.layout = meta.layout
        record.n_samples = meta.n_samples
        inventory_rows.append(record.__dict__)
        metadata_dicts.append(metadata_to_dict(meta))
        metadata_objects.append(meta)

    inventory_df = pd.DataFrame(inventory_rows).sort_values(["split", "relpath"])
    inventory_df["size_gb"] = inventory_df["size_bytes"] / (1024 ** 3)
    inventory_df.to_csv(audit_dir / "file_inventory.csv", index=False)
    save_json(audit_dir / "file_metadata.json", {"files": metadata_dicts})

    tree_rows = []
    for meta in metadata_dicts:
        for row in meta["tree_rows"]:
            tree_rows.append(
                {
                    "file_path": meta["path"],
                    "layout": meta["layout"],
                    "kind": row["kind"],
                    "path": row["path"],
                    "shape": row["shape"],
                    "dtype": row["dtype"],
                }
            )
    pd.DataFrame(tree_rows).to_csv(audit_dir / "dataset_tree.csv", index=False)

    summary = summarize_layouts(metadata_objects)
    summary["total_size_gb"] = float(inventory_df["size_gb"].sum())
    save_json(audit_dir / "audit_summary.json", summary)

    lines: List[str] = []
    lines.append(f"Dataset root: {args.data_root.resolve()}")
    lines.append(f"Files scanned: {len(files)}")
    lines.append(f"Total size: {summary['total_size_gb']:.2f} GiB")
    lines.append(f"Formats: {summary['format_counts']}")
    lines.append(f"Layouts: {summary['layout_counts']}")
    lines.append(f"Split counts: {summary['split_file_counts']}")
    lines.append(f"Declared logical samples: {summary['total_declared_samples']}")
    lines.append(f"Unique alpha values: {summary['unique_alpha_values']}")
    lines.append(f"Unique zeta values: {summary['unique_zeta_values']}")
    lines.append(f"Unique alpha-zeta combinations: {summary['num_alpha_zeta_combinations']}")
    lines.append("")
    lines.append("Representative file structures:")
    for meta in metadata_objects[: min(5, len(metadata_objects))]:
        lines.append(f"- {Path(meta.path).name}: layout={meta.layout}, n_samples={meta.n_samples}, labels={meta.label_info}")
        for field in meta.fields:
            lines.append(f"  {field.dataset_path}: shape={field.shape}, dtype={field.dtype}, channels={field.channel_names}")
    (audit_dir / "structure_audit.txt").write_text("\n".join(lines))

    print(f"Saved audit to {audit_dir}")


if __name__ == "__main__":
    main()
