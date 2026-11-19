"""
Metal records, graph bundles and the train/val split.

A "metal record" is a dict of light metadata about one metal center (refcode,
label, charge, coord_nr, ...). Raw records also carry the live cell2mol
`metal` object; graph records replace it with the built PyG `graph`, which
is what bundles store -- self-contained tensors, ~78x smaller than raw
records, loadable without cell2mol.

Only `collect_records_for_refcode` / `collect_graph_records` touch cell2mol
(imported lazily), so training from a bundle stays cell2mol-free.
"""

from __future__ import annotations

import os
import pickle
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, train_test_split

from .compat import contains, is_metal_complex, load_reference, rel_metal_radius
from .constants import CLASS_TO_OX, OX_STATES

# =============================================================================
# Collecting records from .cell files
# =============================================================================


def cell_path(cell_dir: str | Path, refcode: str) -> str:
    return f"{cell_dir}/{refcode[0]}/{refcode}/Ref_Cell_{refcode}.cell"


def collect_records_for_refcode(
    refcode: str, cell_dir: str | Path, verbose: bool = False
) -> list[dict]:
    """Load one Reference .cell file and return a raw record per metal in
    each of its complexes."""
    try:
        refcell = load_reference(cell_path(cell_dir, refcode))
    except FileNotFoundError:
        print(f"{refcode}: .cell file not found, skipping", flush=True)
        return []

    records = []

    for mol_idx, mol in enumerate(refcell.refmoleclist):
        if not is_metal_complex(mol):
            continue

        for metal_idx, metal in enumerate(mol.metals):
            coordinating_atoms_info = [
                (atom.label, atom.charge) for atom in (metal.coord_sphere_atoms or [])
            ]
            ligands_info = [
                (lig.formula, lig.totcharge, lig.smiles, lig.denticity, lig.haptic_type)
                for lig in mol.ligands
                if contains(lig.metals, metal)
            ]

            if verbose:
                print(refcode, mol.formula, mol.totcharge)
                print(
                    "  ",
                    metal.label,
                    metal.charge,
                    coordinating_atoms_info,
                    ligands_info,
                    rel_metal_radius(metal),
                )

            records.append(
                {
                    "refcode": refcode,
                    "mol_idx": mol_idx,
                    "metal_idx": metal_idx,
                    "metal": metal,
                    "metal_label": metal.label,
                    "metal_charge": metal.charge,
                    "mol_formula": mol.formula,
                    "mol_charge": mol.totcharge,
                    "coordinating_atoms_info": coordinating_atoms_info,
                    "ligands_info": ligands_info,
                    "rel_metal_radius": rel_metal_radius(metal),
                    "stratify_label": f"{metal.label}_{metal.charge}",
                    "coord_nr": metal.coord_nr,
                    "coord_geometry": metal.coord_geometry,
                    "geom_deviation": metal.geom_deviation,
                    "mol_n_metals": len(mol.metals),
                }
            )

    return records


def _graph_records_for_refcode(
    refcode: str, cell_dir: str, variant_names: list[str], verbose: bool
) -> dict[str, list[dict]]:
    """Load one .cell file once and build every requested variant's graphs
    from it -- loading is >99% of the cost, building is cheap."""
    from .variants import make_builder

    builders = {name: make_builder(name) for name in variant_names}
    graph_records = {name: [] for name in variant_names}

    for record in collect_records_for_refcode(refcode, cell_dir, verbose):
        metal = record.pop("metal")
        for name, builder in builders.items():
            graph_records[name].append(
                {**record, "graph": builder.build(metal, oxidation_state=metal.charge)}
            )

    return graph_records


def _graph_records_for_refcode_pickled(
    refcode: str, cell_dir: str, variant_names: list[str], verbose: bool
) -> bytes:
    # Returned pre-pickled: live tensors crossing the process boundary go
    # through PyTorch's shared-memory tensor IPC, which runs out of fds /
    # shm mmap slots at this task count (hit at ~250 and ~8,000 refcodes).
    return pickle.dumps(
        _graph_records_for_refcode(refcode, cell_dir, variant_names, verbose),
        protocol=pickle.HIGHEST_PROTOCOL,
    )


def default_num_workers() -> int:
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def collect_graph_records(
    refcodes,
    cell_dir: str | Path,
    variant_names: list[str],
    num_workers: int = 1,
    verbose: bool = False,
) -> dict[str, list[dict]]:
    """
    Load .cell files and build one graph per metal for each variant in
    `variant_names`; returns {variant: records}. Each worker does load *and*
    build and ships back only the small graph records. Every variant's list
    has the same record order (completion order when num_workers > 1), so
    bundles collected together get identical train/val splits.
    """
    cell_dir = str(cell_dir)
    records = {name: [] for name in variant_names}

    def add(result: dict[str, list[dict]]) -> None:
        for name in variant_names:
            records[name].extend(result[name])

    if num_workers <= 1:
        for refcode in refcodes:
            add(_graph_records_for_refcode(refcode, cell_dir, variant_names, verbose))
        return records

    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        futures = [
            pool.submit(
                _graph_records_for_refcode_pickled,
                refcode,
                cell_dir,
                variant_names,
                verbose,
            )
            for refcode in refcodes
        ]

        for i, future in enumerate(as_completed(futures), start=1):
            add(pickle.loads(future.result()))
            if i % 1000 == 0 or i == len(futures):
                print(f"  processed {i}/{len(futures)} refcodes", flush=True)

    return records


def load_refcodes(path: str | Path) -> np.ndarray:
    return np.atleast_1d(np.loadtxt(path, dtype=str))


# =============================================================================
# Filters
# =============================================================================


def filter_supported_charges(records: list[dict]) -> list[dict]:
    """Drop records whose oxidation state is unassigned (charge=None from a
    failed charge assignment) or outside OX_STATES."""
    dropped = [r for r in records if r["metal_charge"] not in OX_STATES]
    if dropped:
        print(
            f"\nDropping {len(dropped)} metal centers with unsupported/unassigned charge (not in {OX_STATES})."
        )
    return [r for r in records if r["metal_charge"] in OX_STATES]


def filter_coordinated_metals(records: list[dict]) -> list[dict]:
    """
    Drop records with no resolved coordination sphere (coord_nr in (None, 0)),
    e.g. an isolated ion: nothing to learn from, and an edgeless graph crashes
    HeteroConv. Seen once in 108,457 records (isolated Zn2+ in QARNIK).
    """
    dropped = [r["refcode"] for r in records if not r["coord_nr"]]
    if dropped:
        print(
            f"\nDropping {len(dropped)} metal centers with no resolved coordination sphere "
            f"(coord_nr=0): {dropped[:10]}{'...' if len(dropped) > 10 else ''}"
        )
    return [r for r in records if r["coord_nr"]]


# =============================================================================
# Bundles
# =============================================================================


def save_graph_bundle(
    graph_records: list[dict], variant_name: str, node_dims: dict[str, int], path
) -> None:
    with open(path, "wb") as f:
        pickle.dump(
            {
                "graph_records": graph_records,
                "variant": variant_name,
                **{f"{nt}_dim": dim for nt, dim in node_dims.items()},
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


def load_graph_bundle(path) -> dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)


def bundle_node_dims(bundle: dict[str, Any], node_types) -> dict[str, int]:
    missing = [nt for nt in node_types if f"{nt}_dim" not in bundle]
    if missing:
        raise ValueError(f"Bundle has no dims for node types {missing}")
    return {nt: bundle[f"{nt}_dim"] for nt in node_types}


# =============================================================================
# Split + reporting
# =============================================================================


def split_records(
    records: list[dict],
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    group_by: str | None = "refcode",
):
    """
    Train/val split stratified on "<metal>_<charge>" (labels with a single
    sample pooled into "Other"). With group_by="refcode", all metals of one
    refcode land on the same side: the metals of a polynuclear complex share
    most of their environment, so splitting them would leak val into train.
    group_by=None splits per metal record (the behavior of runs before
    2026-10-27).
    """
    raw_counts = Counter(r["stratify_label"] for r in records)
    labels = [
        r["stratify_label"] if raw_counts[r["stratify_label"]] >= 2 else "Other"
        for r in records
    ]
    counts = Counter(labels)
    can_stratify = len(counts) > 1 and min(counts.values()) >= 2

    if group_by is None:
        print(
            "Per-record split, "
            + ("stratified on metal_label + metal_charge" if can_stratify else "random")
        )
        return train_test_split(
            records,
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
            stratify=labels if can_stratify else None,
        )

    groups = [r[group_by] for r in records]
    n_splits = max(2, round(1 / test_size))
    if can_stratify:
        print(f"Split grouped by {group_by}, stratified on metal_label + metal_charge")
        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )
    else:
        print(f"Split grouped by {group_by} (stratification not possible)")
        splitter = GroupKFold(n_splits=n_splits)
    train_idx, val_idx = next(splitter.split(records, labels, groups))

    return [records[i] for i in train_idx], [records[i] for i in val_idx]


def nuclearity_bucket(record: dict) -> str:
    n = record.get("mol_n_metals")
    return {1: "mono", 2: "bi", 3: "tri", 4: "tetra"}.get(
        n, "unknown" if n is None else f"{n}-nuclear"
    )


def print_accuracy_by_nuclearity(
    metrics: dict[str, Any], records: list[dict], title: str
) -> None:
    """Accuracy per nuclearity of each record's molecule (needs mol_n_metals,
    stored by bundles collected since 2026-10-27)."""
    if not records or "mol_n_metals" not in records[0]:
        return
    hits: dict[str, list[bool]] = {}
    for r, t, p in zip(records, metrics["y_true"], metrics["y_pred"]):
        hits.setdefault(nuclearity_bucket(r), []).append(t == p)

    order = {"mono": 0, "bi": 1, "tri": 2, "tetra": 3}
    print(f"\n{title}")
    for bucket in sorted(hits, key=lambda b: order.get(b, 99)):
        h = hits[bucket]
        print(f"  {bucket:12s} n={len(h):6d}  acc={sum(h) / len(h):.4f}")


def print_distribution(records: list[dict], title: str) -> None:
    print(f"\n{title}\n" + "-" * len(title))
    for label, count in Counter(r["stratify_label"] for r in records).most_common():
        print(f"{label:10s} {count}")


def save_split_metadata(
    train_records: list[dict], val_records: list[dict], out_dir: Path
) -> None:
    for name, records in (("train", train_records), ("val", val_records)):
        np.savetxt(
            out_dir / f"{name}_refcodes.txt", [r["refcode"] for r in records], fmt="%s"
        )

        with open(out_dir / f"{name}_metal_records.txt", "w") as f:
            for r in records:
                f.write(
                    f"{r['refcode']} mol_idx={r['mol_idx']} metal_idx={r['metal_idx']} "
                    f"{r['metal_label']} charge={r['metal_charge']} formula={r['mol_formula']}\n"
                )


def write_validation_report(
    metrics: dict[str, Any], val_records: list[dict], csv_path: Path
) -> list:
    """Write per-sample val predictions to `csv_path`; print the errors."""
    errors = []

    with open(csv_path, "w") as f:
        f.write("refcode,metal_label,formula,true_ox,pred_ox,confidence,correct\n")

        for r, t, p, c in zip(
            val_records, metrics["y_true"], metrics["y_pred"], metrics["confidence"]
        ):
            true_ox, pred_ox = CLASS_TO_OX[t], CLASS_TO_OX[p]
            f.write(
                f"{r['refcode']},{r['metal_label']},{r['mol_formula']},{true_ox},{pred_ox},{c:.4f},{t == p}\n"
            )
            if t != p:
                errors.append((r, true_ox, pred_ox, c))

    print(
        f"\nValidation errors: {len(errors)}/{len(val_records)} "
        f"(acc={metrics['accuracy']:.4f}, macro_f1={metrics['macro_f1']:.4f}, loss={metrics['loss']:.4f})"
    )
    print(f"Full per-sample predictions written to {csv_path}\n")

    for r, true_ox, pred_ox, c in errors:
        print(
            f"  {r['refcode']:12s} {r['metal_label']:3s} formula={r['mol_formula']:20s} "
            f"true={true_ox} pred={pred_ox} confidence={c:.3f}"
        )

    return errors
