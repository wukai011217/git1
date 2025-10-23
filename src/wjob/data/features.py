from __future__ import annotations
# -*- coding: utf-8 -*-
"""Feature generation utilities for SOAP descriptors.

This module provides helper functions to generate SOAP features centred on a
specified atom type and to produce a metadata table that maps every feature
index to its corresponding quantum numbers (n1, n2, l) and element pair. The
implementation relies on ``dscribe`` and ``ase`` which are already part of the
project dependencies via ``wjob``.

Typical usage (executed as a CLI)::

    python -m wjob.data.features /path/to/structure/dir

The script will recursively search for VASP ``CONTCAR`` files, determine the
centre element from the third-to-last path component (``.../<elem>/<type>/CONTCAR``)
and save two files next to the structure:

    soap_<elem>.npy          – the raw feature matrix (n_centres × n_features)
    soap_<elem>_meta.csv     – a CSV describing every feature column
"""

import os
import json
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
from ase.io import read
from ase import data as ase_data
from dscribe.descriptors import SOAP

from wjob.config import SOAP_PARAMS

__all__ = [
    "generate_soap",
    "get_feature_metadata",
    "save_soap_features_with_metadata",
    "main",
]


def _ensure_dir(p: Path) -> None:
    """Create parent directory of *p* if it does not already exist."""
    p.parent.mkdir(parents=True, exist_ok=True) 


def generate_soap(structure_file: str | Path, centre_element: str) -> Tuple[np.ndarray, SOAP]:
    """Generate SOAP features for all atoms of *centre_element*.

    Parameters
    ----------
    structure_file
        Path to a VASP ``CONTCAR`` (or POSCAR-like) file.
    centre_element
        Chemical symbol of atoms to be used as centres.

    Returns
    -------
    features
        ``(n_centres, n_features)`` float array.
    soap
        The configured :class:`dscribe.descriptors.SOAP` instance used to create
        *features*. This object is returned so that the caller can query
        additional information (e.g. species list).
    """
    structure_file = Path(structure_file)
    atoms = read(structure_file.as_posix())

    # Build SOAP descriptor with species present in the structure
    unique_species = sorted(set(atoms.get_atomic_numbers()))
    soap = SOAP(species=unique_species, **SOAP_PARAMS)

    # Identify centre indices corresponding to the requested element
    centre_indices = [idx for idx, sym in enumerate(atoms.get_chemical_symbols()) if sym == centre_element]
    if not centre_indices:
        raise ValueError(
            f"Element '{centre_element}' not found in {structure_file}. "
            "Cannot compute centred SOAP features.")

    # DScribe uses the American spelling "centers" for the kwarg name.
    features = soap.create(atoms, centers=centre_indices)  # type: ignore[arg-type]
    return features, soap


def get_feature_metadata(soap: SOAP) -> pd.DataFrame:
    """Construct a dataframe mapping every feature column to (elem1, elem2, n1, n2, l).

    The implementation follows the same ordering rules as DScribe when
    ``compression='off'``. For the current project we assume this default. If
    other compression modes are required in the future, this function will need
    to be extended accordingly.
    """
    if soap.compression["mode"] != "off":
        raise NotImplementedError("Feature metadata generation only implemented for compression='off'.")

    species = list(soap._atomic_numbers)  # internal ordering used by SOAP
    n_elem = len(species)
    n_max = soap._n_max
    l_max = soap._l_max

    records = []
    # Iterate over species pairs (i ≤ j) to match DScribe's ordering
    for i, z1 in enumerate(species):
        sym1 = ase_data.chemical_symbols[z1]
        for j, z2 in enumerate(species[i:], start=i):
            sym2 = ase_data.chemical_symbols[z2]
            if i == j:
                # Symmetric block – only n1 ≤ n2
                for n1 in range(n_max):
                    for n2 in range(n1, n_max):
                        for l in range(l_max + 1):
                            records.append({
                                "elem1": sym1,
                                "elem2": sym2,
                                "n1": n1,
                                "n2": n2,
                                "l": l,
                            })
            else:
                # Full unsymmetric block
                for n1 in range(n_max):
                    for n2 in range(n_max):
                        for l in range(l_max + 1):
                            records.append({
                                "elem1": sym1,
                                "elem2": sym2,
                                "n1": n1,
                                "n2": n2,
                                "l": l,
                            })
    meta_df = pd.DataFrame.from_records(records)

    # Sanity check: number of generated rows must equal soap.get_number_of_features()
    n_features_expected = soap.get_number_of_features()
    if len(meta_df) != n_features_expected:
        raise RuntimeError(
            "Internal error while forming metadata: "
            f"expected {n_features_expected} rows, got {len(meta_df)}.")
    return meta_df


def save_soap_features_with_metadata(
    structure_file: str | Path,
    centre_atom: str,
    output_dir: str | Path | None = None,
) -> Tuple[Path, Path]:
    """Compute SOAP features and save them alongside a metadata CSV.

    Parameters
    ----------
    structure_file
        Path to a VASP ``CONTCAR``.
    centre_atom
        Chemical symbol of the centre atoms.
    output_dir
        Directory where output files should be written. Defaults to the same
        directory as *structure_file*.

    Returns
    -------
    feature_path, meta_path
        The filesystem paths of the written ``.npy`` and ``.csv`` files.
    """
    structure_file = Path(structure_file)
    if output_dir is None:
        # 将路径中的'raw'替换为'fea'
        path_str = str(structure_file.parent)
        if 'raw' in path_str:
            output_dir = Path(path_str.replace('raw', 'fea'))
        else:
            output_dir = structure_file.parent
        # 确保输出目录存在
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_dir = Path(output_dir)

    # Generate SOAP features and metadata
    print(structure_file, centre_atom)
    features, soap = generate_soap(structure_file, centre_atom)
    meta_df = get_feature_metadata(soap)
    features_df = pd.DataFrame(features)

    # Prepare output paths
    feat_path = output_dir / f"soap_{centre_atom}.npy"
    feat_pathcsv = output_dir / f"soap_{centre_atom}.csv"
    meta_path = output_dir / f"soap_{centre_atom}_meta.csv"
    _ensure_dir(feat_path)

    # Persist to disk
    print(features.shape)
    np.save(feat_path, features)
    meta_df.to_csv(meta_path, index=False)
    features_df.to_csv(feat_pathcsv, index=False)

    print(f"Saved SOAP features to {feat_path}")
    print(f"Saved feature metadata to {meta_path}")
    return feat_path, meta_path


# --------------------------------------------------------------------------------------
# CLI helper – replicates earlier prototype
# --------------------------------------------------------------------------------------

def main(structure_dir: str, centre_element: str = None, centre_index: int = None) -> None:
    """Recursively search *structure_dir* for CONTCAR files and process them.
    
    Parameters
    ----------
    structure_dir
        Directory to search for CONTCAR files.
    centre_element
        Chemical symbol of the centre atoms. If None, will be extracted from the path.
    centre_index
        Index of the atom to use as centre. If provided, overrides centre_element.
    """
    for root, _dirs, files in os.walk(structure_dir):
        if "CONTCAR" not in files:
            continue
        structure_file = Path(root) / "CONTCAR"
        
        # 确定中心原子
        current_centre_element = centre_element
        
        # 如果没有指定中心原子，从路径中提取（默认方式）
        if current_centre_element is None and centre_index is None:
            # Extract centre element from the third-to-last path component
            path_parts = Path(root).parts
            if len(path_parts) < 3:
                print(f"[WARN] Path too short to determine centre element: {root}")
                continue
            current_centre_element = path_parts[-3]
            print(f"Processing {structure_file} (centre={current_centre_element} from path)")
        
        # 如果指定了中心原子索引，则使用索引方式
        elif centre_index is not None:
            # 使用索引指定中心原子，需要读取结构文件获取原子类型
            try:
                from ase.io import read
                atoms = read(structure_file.as_posix())
                if centre_index >= len(atoms):
                    print(f"[ERROR] Centre index {centre_index} out of range for {structure_file} with {len(atoms)} atoms")
                    continue
                current_centre_element = atoms.get_chemical_symbols()[centre_index]
                print(f"Processing {structure_file} (centre={current_centre_element} from index {centre_index})")
            except Exception as exc:
                print(f"[ERROR] Failed to read structure file {structure_file}: {exc}")
                continue
        else:
            # 使用指定的中心原子
            print(f"Processing {structure_file} (centre={current_centre_element} as specified)")

        try:
            save_soap_features_with_metadata(structure_file, current_centre_element)
        except Exception as exc:
            print(f"[ERROR] Failed to process {structure_file}: {exc}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate SOAP features for structures")
    parser.add_argument("structure_dir", type=str, help="Directory containing structure files")
    parser.add_argument("--centre", "-c", type=str, help="Centre element symbol (overrides path extraction)")
    parser.add_argument("--index", "-i", type=int, help="Atom index to use as centre (overrides centre element)")
    
    args = parser.parse_args()
    
    main(args.structure_dir, centre_element=args.centre, centre_index=args.index)