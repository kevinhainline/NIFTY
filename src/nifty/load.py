"""Load observed data and filters from various file formats."""

import json
from collections.abc import Sequence
from pathlib import Path

import fitsio
import numpy as np
import numpy.typing as npt
import pandas as pd
from numpy.lib.recfunctions import structured_to_unstructured

__all__ = [
    "load_filter_info",
    "load_phot_catalog_fits",
    "load_phot_catalog_text",
    "load_spec_text",
]


def load_phot_catalog_text(
    path: str | Path,
    filter_desc: Sequence[dict[str, str | int]],
    ids: Sequence[int],
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Read a photometry catalog from text.

    The text file must be whitespace-delimited with one row per object. It must
    have an ID column and columns matching the column names described in
    filter_desc.

    Parameters
    ----------
    path : str or Path
        Path to photometry text file.
    filter_desc : sequence of dict
        Sequence of filter descriptions to use in order. Each description must
        have the keys `"flux"` and `"error"` for the flux and error columns
        respectively.
    ids : sequence of int
        Object IDs to load from the file.

    Returns
    -------
    flux : ndarray of shape (N, F)
        Flux values for each of the N objects through F filters.
    error : ndarray of shape (N, F)
        Flux errors for each of the N objects through F filters.
    """
    data = pd.read_csv(path, sep=r"\s+", engine="c", comment="#")
    data = data.set_index("ID").loc[ids].reset_index(drop=True)
    flux_cols = [desc["flux"] for desc in filter_desc]
    error_cols = [desc["error"] for desc in filter_desc]
    flux = data.loc[:, flux_cols].to_numpy(dtype=np.float32)
    error = data.loc[:, error_cols].to_numpy(dtype=np.float32)
    return flux, error


def load_phot_catalog_fits(
    path: str | Path,
    filter_desc: Sequence[dict[str, str | int]],
    ids: Sequence[int],
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Read a photometry catalog from a FITS file.

    The FITS file must have HDU extensions and column names matching filter_desc
    and an ID column in each HDU extension with filter fluxes.

    Parameters
    ----------
    path : str or Path
        Path to photometry FITS file.
    filter_desc : sequence of dict
        Sequence of filter descriptions to use in order. Each description must
        have the key `"extension"` for the extension index or name with that
        filter and the keys `"flux"` and `"error"` for the flux and error
        columns respectively.
    ids : sequence of int
        Object IDs to load from the file.

    Returns
    -------
    flux : ndarray of shape (N, F)
        Flux values for each of the N objects through F filters.
    error : ndarray of shape (N, F)
        Flux errors for each of the N objects through F filters.
    """
    # Map extension name to a tuple of lists for index, flux, and error.
    hdus = {}
    for i, col_desc in enumerate(filter_desc):
        ext = col_desc["extension"]
        if ext not in hdus:
            hdus[ext] = ([], [], [])
        hdu = hdus[ext]
        hdu[0].append(i)
        hdu[1].append(col_desc["flux"])
        hdu[2].append(col_desc["error"])

    target_ids = np.array(ids)
    flux = np.empty(target_ids.shape + (len(filter_desc),), dtype=np.float32)
    error = np.empty(target_ids.shape + (len(filter_desc),), dtype=np.float32)
    with fitsio.FITS(path) as f:
        for extension, (col_i, flux_cols, error_cols) in hdus.items():
            hdu = f[extension]
            id_col = hdu.read_column("ID")
            # The indices of the rows containing the object IDs.
            sorter = np.argsort(id_col)
            row_idx = np.searchsorted(id_col, target_ids, sorter=sorter)
            found = np.zeros(len(target_ids), dtype=bool)
            valid = row_idx < len(id_col)
            found[valid] = id_col[sorter[row_idx[valid]]] == target_ids[valid]
            if not np.all(found):
                raise KeyError(
                    f"Not all IDs were found in {extension}. "
                    f"Missing IDs: {target_ids[~found]}."
                )
            rows = sorter[row_idx]
            # Concatenate the lists of columns to get all columns.
            all_cols = flux_cols + error_cols
            data = hdu.read(rows=rows, columns=all_cols)
            for arr, cols in ((flux, flux_cols), (error, error_cols)):
                arr[:, col_i] = structured_to_unstructured(
                    data[cols], dtype=np.float32
                )
    return flux, error


def load_spec_text(
    path: str | Path,
) -> tuple[
    npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]
]:
    """Read a spectrum text file.

    The text file must be formatted with three whitespace-delimited columns:
    wavelength in Angstroms, flux in nJy, and flux error in nJy.

    Parameters
    ----------
    path : str or Path
        Path to the spectrum text file.

    Returns
    -------
    wave : ndarray
        Wavelength in Angstroms.
    flux : ndarray
        Flux in nJy.
    error : ndarray
        Flux error in nJy.
    """
    data = pd.read_csv(
        path, sep=r"\s+", comment="#", header=None, engine="c", dtype=np.float32
    ).to_numpy()
    data[:, 0] *= 1e4  # Ang
    # Sort by wavelength.
    data = data[data[:, 0].argsort()]
    return data[:, 0], data[:, 1], data[:, 2]


def load_filter_info(path: str | Path) -> Sequence[dict[str, str | int]]:
    """Load filter information from a NIFTY configuration file.

    Parameters
    ----------
    path : str or Path
        Path to the NIFTY configuration file to load.

    Returns
    -------
    filter_info : sequence of dict
        Information dictionary for each filter with the key "name" for the
        sedpy filter name and remaining items specified in the filter
        configuration file.

    Notes
    -----
    If the filter name is not prefixed with the telescope name (e.g. "jwst_"),
    it is assumed to be a JWST filter.
    """
    with open(path) as f:
        filter_config = json.load(f)["filter_columns"]
    filters = [
        {
            "name": name.lower() if "_" in name else "jwst_" + name.lower(),
            **load_info,
        }
        for name, load_info in filter_config.items()
    ]
    return filters
