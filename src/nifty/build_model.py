"""Create various model grids from their native formats."""

import io
import json
import re
import tarfile
import tempfile
import zipfile
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import Protocol, Self

import numpy as np
import numpy.typing as npt
import pandas as pd
from sedpy.observate import Filter
from tqdm import tqdm

from .load import load_filter_info
from .model import ModelGrid
from .spectra import flambda_to_fnu, resample_spectrum, trim_wavelength_coverage

__all__ = [
    "build_atmo2020",
    "build_lowz",
    "build_sonora_diamondback_cloudless",
    "build_sonora_diamondback_cloudy",
    "build_sonora_elf_owl",
    "build_sonora_flame_skimmer_disequilibrium",
    "build_sonora_flame_skimmer_equilibrium",
    "build_sonora_flame_skimmer_quickstart",
    "build_sonora_ph3",
]

# Standard lower-resolution wavelength grid for infrared.
WAVE_START = 0.3  # um
WAVE_STOP = 15  # um
WAVE_RES = 1500
LOWER_RES_WAVE = (
    np.geomspace(
        WAVE_START,
        WAVE_STOP,
        int(np.ceil(WAVE_RES * np.log(WAVE_STOP / WAVE_START))) + 1,
    ).astype(np.float32)
    * 1e4
)  # Ang


class _SupportsReadBytes(Protocol):
    def read(self, size: int = -1, /) -> bytes: ...
    def close(self) -> None: ...


class ProgressByteReader:
    """Byte stream that shows a progress bar of the number of bytes read.

    Parameters
    ----------
    fileobj : file-like object
        File-like object that supports reading bytes and closing.
    size : int, optional
        Total file size. If not given, only progress statistics are shown
        without a progress bar.
    **kwargs
        Extra arguments passed to `tqdm`.
    """

    def __init__(
        self,
        fileobj: _SupportsReadBytes,
        size: int | None = None,
        /,
        **kwargs,
    ) -> None:
        kwargs.setdefault("unit", "B")
        kwargs.setdefault("unit_scale", True)
        self._fileobj = fileobj
        self._pbar = tqdm(
            total=size,
            **kwargs,
        )

    @classmethod
    def open(cls, path: str | Path, /, **kwargs) -> Self:
        """Open a file for byte reading with progress.

        Parameters
        ----------
        path : str or Path
            Path to the file.
        **kwargs
            Extra arguments passed to `tqdm`.
        """
        path = Path(path)
        fileobj = open(path, "rb")
        kwargs.setdefault("desc", path.name)
        return cls(fileobj, path.stat().st_size, **kwargs)

    def read(self, size=-1, /) -> bytes:
        data = self._fileobj.read(size)
        self._pbar.update(len(data))
        return data

    def __getattr__(self, name):
        return getattr(self._fileobj, name)

    def close(self) -> None:
        self._fileobj.close()
        self._pbar.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def _ab_mag_to_fnu_cgs(
    ab_mag: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """
    Convert AB magnitude to f_nu in cgs.

    Parameters
    ----------
    ab_mag : ndarray
        AB magnitude value.

    Returns
    -------
    Flux density in f_nu with units erg/s/cm^2/Hz.
    """
    return 10.0 ** ((ab_mag + 48.60) / -2.5)


def load_filters(path: str | Path) -> tuple[list[Filter], list[str]]:
    """
    Load filters from the NIFTY JSON config file.

    Returns
    -------
    filters : list of sedpy.observate.Filter
        Filters loaded from config file.
    filter_names : list of str
        Filter names.
    """
    filter_info = load_filter_info(path)
    filter_names = [str(info["name"]) for info in filter_info]
    filters = [Filter(name) for name in filter_names]
    return filters, filter_names


def _filter_fluxes(
    wave: npt.NDArray[np.float32],
    flux: npt.NDArray[np.float32],
    filters: Sequence[Filter],
) -> npt.NDArray[np.float32]:
    """
    Pass a single spectrum through all filters.

    Parameters
    ----------
    wave : ndarray of shape (W,)
        Wavelength in Angstroms. Must be monotonically increasing.
    flux : ndarray of shape (..., W)
        f_lambda in erg/s/cm^2/Ang in the same order as wave.
    filters : sequence of sedpy.observate.Filter
        Filters that the spectrum is passed through.

    Returns
    -------
    fluxes : np.ndarray of shape (..., n_filters)
        Flux density in f_nu with units erg/s/cm^2/Hz.
    """
    # Replace the last axis with the filters.
    shape = flux.shape[:-1] + (len(filters),)
    fluxes = np.empty(shape, dtype=np.float32)
    for i, f in enumerate(filters):
        fluxes[..., i] = _ab_mag_to_fnu_cgs(f.ab_mag(wave, flux))
    return fluxes


# ============================================================
# Model: Sonora Flame Skimmer
#
# Expected directory layout
# -------------------------
#   <model_path>/spectra_R30k_*_<chem>.zip
#       flameskimmer_R30000_teff_<Tmin>-<Tmax>_<chem>/manifest.csv
#       flameskimmer_R30000_teff_<Tmin>-<Tmax>_<chem>/teff_*/.../*.nc
# ============================================================
def build_sonora_flame_skimmer_disequilibrium(
    path: Path | str,
    *,
    filters: Sequence[Filter],
    filter_names: Sequence[str],
    progress_bar: bool = True,
) -> ModelGrid:
    """Build the raw Sonora Flame Skimmer disequilibrium ModelGrid.

    Parameters
    ----------
    path : Path or str
        Directory containing spectra_R30k_*_deq.zip archives.
    filters : sequence of sedpy.observate.Filter
        Photometric filters.
    filter_names : sequence of str
        Names of photometric filters to store as metadata.
    progress_bar : bool, default=True
        Display a progress bar while loading each archive.

    Returns
    -------
    model_grid : ModelGrid
        Sonora Flame Skimmer ModelGrid with disequilibrium chemistry only.
    """
    return _build_sonora_flame_skimmer(
        path,
        equilibrium=False,
        filters=filters,
        filter_names=filter_names,
        progress_bar=progress_bar,
    )


def build_sonora_flame_skimmer_equilibrium(
    path: Path | str,
    *,
    filters: Sequence[Filter],
    filter_names: Sequence[str],
    progress_bar: bool = True,
) -> ModelGrid:
    """Build the raw Sonora Flame Skimmer equilibrium ModelGrid.

    Parameters
    ----------
    path : Path or str
        Directory containing spectra_R30k_*_eq.zip archives.
    filters : sequence of sedpy.observate.Filter
        Photometric filters.
    filter_names : sequence of str
        Names of photometric filters to store as metadata.
    progress_bar : bool, default=True
        Display a progress bar while loading each archive.

    Returns
    -------
    model_grid : ModelGrid
        Sonora Flame Skimmer ModelGrid with equilibrium chemistry only.
    """
    return _build_sonora_flame_skimmer(
        path,
        equilibrium=True,
        filters=filters,
        filter_names=filter_names,
        progress_bar=progress_bar,
    )


def _build_sonora_flame_skimmer(
    path: Path | str,
    *,
    equilibrium: bool,
    filters: Sequence[Filter],
    filter_names: Sequence[str],
    progress_bar: bool,
) -> ModelGrid:
    try:
        import xarray
    except ImportError:
        raise ImportError(
            "Building models requires the [build-model] optional dependencies. "
            "Install with `pip install astro-nifty[build-model]`"
        )
    path = Path(path)
    chemistry = "eq" if equilibrium else "deq"
    parts = sorted(path.glob(f"spectra_R30k_*_{chemistry}.zip"))

    values = []
    model_wave = None
    new_wave = LOWER_RES_WAVE
    for part in parts:
        with zipfile.ZipFile(part) as zip_ref:
            manifest_paths = [
                PurePosixPath(info.filename)
                for info in zip_ref.infolist()
                if not info.is_dir()
                and PurePosixPath(info.filename).name == "manifest.csv"
            ]
            if len(manifest_paths) != 1:
                raise ValueError(
                    f"{part}: expected one manifest.csv, found {len(manifest_paths)}"
                )
            parent_path = manifest_paths[0].parent
            with zip_ref.open(str(manifest_paths[0])) as f:
                index = pd.read_csv(f)
            index = index[index["chem"] == chemistry]
            for relative_path, teff, grav, logkzz, mh, co in tqdm(
                zip(
                    index["relative_path"],
                    index["teff"],
                    index["grav"],
                    index["logkzz"],
                    index["mh"],
                    index["co"],
                ),
                total=len(index),
                desc=part.name,
                disable=not progress_bar,
            ):
                input_file = zip_ref.open(str(parent_path / relative_path))
                with input_file as f:
                    data = xarray.load_dataset(f, engine="h5netcdf")
                params = (
                    np.log10(teff),
                    np.log10(grav) + 2.0,  # cgs
                    *((logkzz,) if not equilibrium else ()),
                    mh,
                    co * 0.458,  # absolute C/O
                )
                if model_wave is None:
                    model_wave = data["wavelength"].values * 1e4  # Ang
                    assert np.all(model_wave[1:] >= model_wave[:-1])
                    new_wave = trim_wavelength_coverage(
                        LOWER_RES_WAVE, model_wave
                    )
                flux = data["flux_emission"].values * 1e-8  # erg/s/cm^2/Ang
                arrays = (
                    _filter_fluxes(model_wave, flux, filters)
                    * 1e23
                    * 1e9,  # nJy
                    flambda_to_fnu(
                        new_wave,
                        resample_spectrum(new_wave, model_wave, flux),
                    )
                    * 1e23
                    * 1e9,  # nJy
                )
                values.append((params, arrays))
    if not values:
        raise ValueError(f"No {chemistry} spectra found in manifests in {path}")
    return _build_from_values(
        values,
        filter_names=filter_names,
        axes=["teff", "logg", *([] if equilibrium else ["kzz"]), "mh", "co"],
        model_name=(
            "SonoraFlameSkimmerEq" if equilibrium else "SonoraFlameSkimmerDeq"
        ),
        wave=new_wave,
    )


# ============================================================
# Model: Sonora Flame Skimmer (Quickstart)
#
# Expected layout
# -------------------------
#   spectra_R3k_quickstart.zip
#      R3000_starter_spectra_manifest.csv
#      deq/mh_+0.0/co_0.5/spectra_logkzz_2_teff_1000_grav_1000_mh_+0.0_co_0.5_R3000.nc
#      ...
# ============================================================
def build_sonora_flame_skimmer_quickstart(
    path: Path | str,
    *,
    filters: Sequence[Filter],
    filter_names: Sequence[str],
    progress_bar: bool = True,
) -> ModelGrid:
    """Build the raw Sonora Flame Skimmer Quickstart Model.

    Parameters
    ----------
    path : Path or str
        Path to the R3k quickstart Sonora Flame Skimmer models.
    filters : sequence of sedpy.observate.Filter
        Photometric filters.
    filter_names : sequence of str
        Names of photometric filters to store as metadata.
    progress_bar : bool, default=True
        Display a progress bar while loading the model.

    Returns
    -------
    model_grid : ModelGrid
        Sonora Flame Skimmer Quickstart ModelGrid object.

    Notes
    -----
    The quickstart models are sampled on a sparser grid than the full models and
    include lower resolution spectra. This reduced size makes them significantly
    easier to process (The full models cover ~150 GB whereas the quickstart
    models are ~9 GB). For NIFTY, the lower resolution spectra negligibly affect
    photometry and spectroscopy. In some cases, the sparser grid will lead to
    greater interpolation errors.

    This function only builds the disequilibrium chemistry models, ignoring the
    equilibrium models.
    """
    try:
        import xarray
    except ImportError:
        raise ImportError(
            "Building models requires the [build-model] optional dependencies. "
            "Install with `pip install astro-nifty[build-model]`"
        )
    values = []
    model_wave = None
    new_wave = LOWER_RES_WAVE
    with zipfile.ZipFile(path) as zip_ref:
        with zip_ref.open("R3000_starter_spectra_manifest.csv") as f:
            index = pd.read_csv(f)
            index = index[index["chem"] == "deq"]
        for path, *params in tqdm(
            zip(
                index["relative_path"],
                index["teff"],
                index["grav"],
                index["logkzz"],
                index["mh"],
                index["co"],
            ),
            disable=not progress_bar,
            total=len(index),
        ):
            with zip_ref.open(path) as f:
                data = xarray.load_dataset(f, engine="h5netcdf")
            params = (
                np.log10(params[0]),
                np.log10(params[1]) + 2.0,  # cgs
                params[2],
                params[3],
                # Normalize to absolute C/O based on Flame Skimmer's solar C/O.
                params[4] * 0.458,
            )
            flux = data["flux_emission"].values * 1e-8  # erg/s/cm^2/Ang
            flux = flux[::-1]
            if model_wave is None:
                model_wave = data["wavelength"].values * 1e4  # Ang
                model_wave = model_wave[::-1]
                assert np.all(model_wave[1:] >= model_wave[:-1])
                new_wave = trim_wavelength_coverage(LOWER_RES_WAVE, model_wave)
            arrays = (
                _filter_fluxes(model_wave, flux, filters) * 1e23 * 1e9,  # nJy
                flambda_to_fnu(
                    new_wave,  # Ang
                    resample_spectrum(new_wave, model_wave, flux),
                )
                * 1e23
                * 1e9,  # nJy
            )
            values.append((params, arrays))
    return _build_from_values(
        values,
        filter_names=filter_names,
        axes=["teff", "logg", "kzz", "mh", "co"],
        model_name="SonoraFlameSkimmerQuickstart",
        wave=new_wave,
    )


# ============================================================
# Model: Sonora Diamondback
#
# Expected layout
# -------------------------
#   diamondback-spectra.zip
#       spectra/t1300g3160f1_m0.0_co1.0.spec
#       spectra/t1600g31nc_m-0.5_co1.0.spec
#       ...
#
# The first three lines are headers. Data columns are wavelength in microns
# (descending) and flux in W/m^2/m.
# ============================================================
_DIAMONDBACK_FILENAME = re.compile(
    r"t(?P<teff>\d+)g(?P<grav>\d+)(?P<cloud>nc|f\d+)"
    r"_m(?P<mh>[+-]?\d+(?:\.\d+)?)_co\d+(?:\.\d+)?\.spec"
)


def _build_sonora_diamondback(
    path: Path | str,
    *,
    cloudy: bool,
    filters: Sequence[Filter],
    filter_names: Sequence[str],
    progress_bar: bool,
) -> ModelGrid:
    model_wave = None
    values = []
    new_wave = LOWER_RES_WAVE
    with zipfile.ZipFile(path) as archive:
        members = [
            (name, match)
            for name in archive.namelist()
            if Path(name).parent == Path("spectra")
            and (match := _DIAMONDBACK_FILENAME.fullmatch(Path(name).name))
            and (match.group("cloud") != "nc") == cloudy
        ]
        for name, match in tqdm(members, disable=not progress_bar):
            with archive.open(name) as spectrum:
                data = np.loadtxt(
                    spectrum, skiprows=3, usecols=(0, 1), dtype=np.float32
                )
            if model_wave is None:
                model_wave = data[:, 0] * 1e4  # Ang
                model_wave = model_wave[::-1]
                assert np.all(model_wave[1:] >= model_wave[:-1])
                new_wave = trim_wavelength_coverage(LOWER_RES_WAVE, model_wave)
            flux = data[:, 1] * 1e-7  # erg/s/cm^2/Ang
            flux = flux[::-1]
            params = (
                np.log10(float(match.group("teff"))),
                round(np.log10(float(match.group("grav"))) + 2, 1),  # cgs
                float(match.group("mh")),
            )
            if cloudy:
                params += (float(match.group("cloud")[1:]),)
            arrays = (
                _filter_fluxes(model_wave, flux, filters) * 1e23 * 1e9,  # nJy
                flambda_to_fnu(
                    new_wave,
                    resample_spectrum(new_wave, model_wave, flux),
                )
                * 1e23
                * 1e9,  # nJy
            )
            values.append((params, arrays))
    return _build_from_values(
        values,
        filter_names=filter_names,
        axes=["teff", "logg", "mh", "fsed"]
        if cloudy
        else ["teff", "logg", "mh"],
        model_name=(
            "SonoraDiamondbackCloudy"
            if cloudy
            else "SonoraDiamondbackCloudless"
        ),
        wave=new_wave,
    )


def build_sonora_diamondback_cloudy(
    path: Path | str,
    *,
    filters: Sequence[Filter],
    filter_names: Sequence[str],
    progress_bar: bool = True,
) -> ModelGrid:
    """Build the raw cloudy Sonora Diamondback ModelGrid.

    Parameters
    ----------
    path : Path or str
        Path to Sonora Diamondback spectra ZIP file.
    filters : sequence of sedpy.observate.Filter
        Photometric filters.
    filter_names : sequence of str
        Names of photometric filters to store as metadata.
    progress_bar : bool, default=True
        Display a progress bar while loading the model.

    Returns
    -------
    model_grid : ModelGrid
        Cloudy Sonora Diamondback ModelGrid object.
    """
    return _build_sonora_diamondback(
        path,
        cloudy=True,
        filters=filters,
        filter_names=filter_names,
        progress_bar=progress_bar,
    )


def build_sonora_diamondback_cloudless(
    path: Path | str,
    *,
    filters: Sequence[Filter],
    filter_names: Sequence[str],
    progress_bar: bool = True,
) -> ModelGrid:
    """Build the raw cloudless Sonora Diamondback ModelGrid.

    Parameters
    ----------
    path : Path or str
        Path to Sonora Diamondback spectra ZIP file.
    filters : sequence of sedpy.observate.Filter
        Photometric filters.
    filter_names : sequence of str
        Names of photometric filters to store as metadata.
    progress_bar : bool, default=True
        Display a progress bar while loading the model.

    Returns
    -------
    model_grid : ModelGrid
        Cloudless Sonora Diamondback ModelGrid object.
    """
    return _build_sonora_diamondback(
        path,
        cloudy=False,
        filters=filters,
        filter_names=filter_names,
        progress_bar=progress_bar,
    )


# ============================================================
# Model: Sonora Elf Owl v2
#
# Expected directory layout
# -------------------------
#   <model_path>/
#       teff_275_325.tar.gz
#       teff_350_400.tar.gz
#       ...
#
# File format: NetCDF (.nc), opened with xarray.
#   ds["wavelength"] : microns
#   ds["flux"]       : erg/s/cm^2/cm  (f_lambda per cm, not per Ang)
# ============================================================
def build_sonora_elf_owl(
    path: Path | str,
    *,
    filters: Sequence[Filter],
    filter_names: Sequence[str],
    progress_bar: bool = True,
) -> ModelGrid:
    """Build the raw Sonora Elf Owl v2 ModelGrid.

    Parameters
    ----------
    path : Path or str
        Path to directory containing teff_<Tmin>_<Tmax>.tar.gz subdirectories.
    filters : sequence of sedpy.observate.Filter
        Photometric filters.
    filter_names : sequence of str
        Names of photometric filters to store as metadata.
    progress_bar : bool, default=True
        Display a progress bar while loading the model.

    Returns
    -------
    model_grid : ModelGrid
        Sonora Elf Owl ModelGrid object.
    """
    try:
        import xarray
    except ImportError:
        raise ImportError(
            "Building models requires the [build-model] optional dependencies. "
            "Install with `pip install astro-nifty[build-model]`"
        )
    path = Path(path)
    model_wave = None
    new_wave = LOWER_RES_WAVE
    values = []
    for part in path.iterdir():
        if not part.suffixes == [".tar", ".gz"]:
            continue
        with (
            ProgressByteReader.open(part) if progress_bar else open(part, "rb")
        ) as fileobj:
            with tarfile.open(fileobj=fileobj, mode="r|gz") as in_tar:
                for member in in_tar:
                    if not member.isfile():
                        continue
                    input_file = in_tar.extractfile(member)
                    if input_file is None:
                        continue
                    file_bytes = input_file.read()
                    stream = io.BytesIO(file_bytes)
                    data = xarray.load_dataset(stream, engine="h5netcdf")
                    all_params = json.loads(data.attrs["planet_params"])
                    params = (
                        np.log10(all_params["teff"]["value"]),
                        np.log10(all_params["logg"]["value"]) + 2.0,  # cgs
                        all_params["logkzz"],
                        all_params["mh"],
                        # Normalize to absolute C/O based on Elf Owl's solar C/O.
                        all_params["cto"] * 0.458,
                    )
                    flux = data["flux"].values * 1e-8  # erg/s/cm^2/Ang
                    flux = flux[::-1]
                    if model_wave is None:
                        model_wave = data["wavelength"].values * 1e4  # Ang
                        model_wave = model_wave[::-1]
                        assert np.all(model_wave[1:] >= model_wave[:-1])
                        new_wave = trim_wavelength_coverage(
                            LOWER_RES_WAVE, model_wave
                        )
                    arrays = (
                        _filter_fluxes(model_wave, flux, filters)
                        * 1e23
                        * 1e9,  # nJy
                        flambda_to_fnu(
                            new_wave,  # Ang
                            resample_spectrum(new_wave, model_wave, flux),
                        )
                        * 1e23
                        * 1e9,  # nJy
                    )
                    values.append((params, arrays))
    return _build_from_values(
        values,
        filter_names=filter_names,
        axes=["teff", "logg", "kzz", "mh", "co"],
        model_name="SonoraElfOwl",
        wave=new_wave,
    )


# ============================================================
# Model: Sonora Elf Owl + PH3
#
# Expected layout
# -------------------------
#   elf_owl_disequilibrium_PH3.npz
#
# File format: NumPy .npz archive.
#   data['Teff']    : 1-D array of Teff values (K)
#   data['logg']    : 1-D array of surface gravity (cgs / 100,
#                     same convention as Sonora Elf Owl)
#   data['logkzz']  : 1-D array of log10(Kzz)
#   data['logmh']   : 1-D array of [M/H]
#   data['cto']     : 1-D array of C/O
#   data['wvno']    : 1-D array of wavenumber (cm^-1)
#                     -> wavelength in microns = 1e4 / wvno
#   data['spectra'] : 5-D array (nTeff, nlogg, nkzz, nmh, nco, nwave)
#                     in erg/s/cm^2/cm
# ============================================================
def build_sonora_ph3(
    path: Path | str,
    *,
    filters: Sequence[Filter],
    filter_names: Sequence[str],
    progress_bar: bool = True,
) -> ModelGrid:
    """Build the raw Sonora Elf Owl + PH3 ModelGrid.

    Parameters
    ----------
    path : Path or str
        Path to the primary NPZ file containing data.
    filters : sequence of sedpy.observate.Filter
        Photometric filters.
    filter_names : sequence of str
        Names of photometric filters to store as metadata.
    progress_bar : bool, default=True
        Display a progress bar while loading the model.

    Returns
    -------
    model_grid : ModelGrid
        Sonora Elf Owl + PH3 ModelGrid object.
    """
    # Load all of the smaller arrays, which can fit into memory.
    # np.load is lazy, so we won't load the big spectra.npy array yet.
    data = np.load(path)
    points = (
        np.log10(data["Teff"]),
        np.log10(data["logg"]) + 2,  # cm/s^2
        data["logkzz"],
        data["logmh"],
        # Normalize to absolute C/O based on Elf Owl's solar C/O.
        data["cto"] * 0.458,
    )
    points = tuple(p.astype(np.float32) for p in points)
    # Wavelength from wave number.
    model_wave = (1e8 / data["wvno"]).astype(np.float32)  # Ang
    model_wave = model_wave[::-1]  # Reverse to sort by wavelength.
    assert np.all(model_wave[1:] >= model_wave[:-1])
    new_wave = trim_wavelength_coverage(LOWER_RES_WAVE, model_wave)
    data.close()
    del data

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / "spectra.npy"
        # Load the spectra array into a temporary file on disk.
        with zipfile.ZipFile(path) as zf:
            with (
                ProgressByteReader(
                    zf.open("spectra.npy", "r"),
                    zf.getinfo("spectra.npy").file_size,
                    desc="copy spectra",
                )
                if progress_bar
                else zf.open("spectra.npy", "r") as src,
                open(temp_path, "wb") as out,
            ):
                while buffer := src.read(1_048_576):
                    out.write(buffer)
        flux = np.load(temp_path, mmap_mode="r")
        original_shape = flux.shape
        # Reshape all but the last dimension to flat for batch indexing.
        flux = flux.reshape((-1, original_shape[-1]))
        phot = np.empty((flux.shape[0], len(filters)), dtype=np.float32)
        spec = np.empty((flux.shape[0], len(new_wave)), dtype=np.float32)
        batch_size = 2000
        for start in tqdm(range(0, flux.shape[0], batch_size), desc="reduce"):
            # Make sure we don't go out of bounds.
            stop = min(start + batch_size, flux.shape[0])
            batch = flux[start:stop].astype(np.float32) * 1e-8  # erg/s/cm^2/Ang
            batch = batch[..., ::-1]
            phot[start:stop] = _filter_fluxes(
                model_wave, batch, filters
            )  # erg/s/cm^2/Hz
            spec[start:stop] = resample_spectrum(
                new_wave, model_wave, batch
            )  # erg/s/cm^2/Ang
        phot = phot.reshape(original_shape[:-1] + (len(filters),))
        spec = spec.reshape(original_shape[:-1] + (len(new_wave),))
        del flux
    phot *= 1e23 * 1e9  # nJy
    spec = (flambda_to_fnu(new_wave, spec) * 1e23 * 1e9).astype(
        np.float32
    )  # nJy
    return ModelGrid(
        points=points,
        phot=phot,
        spec=spec,
        axes=("teff", "logg", "kzz", "mh", "co"),
        model_name="SonoraElfOwlPH3",
        fill_method=None,
        filters=tuple(filter_names),
        wave=new_wave,
    )


# ============================================================
# Model: ATMO2020++
#
# Expected directory layout
# -------------------------
#   <model_path>/
#       grid_m1.0/
#           spec_jwst_t<Teff>_g<logg>_<mh>_kg_g<logg2>.dat
#           ...
#       grid_m0.5/
#       grid_p0/
#       grid_p0.3/
#
# File format: whitespace-delimited ASCII, two columns:
#   col 0: wavelength in microns
#   col 1: flux in W/m^2/um  ->  multiply by 0.1 to get erg/s/cm^2/Ang
#
# Filename format: spec_jwst_t<Teff>_g<logg>_<mh>_kg_g<logg2>.dat
#   split on '_': [0]=spec [1]=jwst [2]=t<Teff> [3]=g<logg> [4]=<mh> ...
#   teff from [2][1:]
#   logg from [3][1:]
#   mh   from [4], prefix 'm' -> negative, 'p' -> positive
#
# Parameters: Teff, logg, [M/H]  (no Kzz or C/O in this grid)
#
# Fluxes are normalized to R = 0.1 Rsun at D = 10 pc.
# Renormalized so that scaling by R^2 / D^2 gives the correct flux.
# ============================================================
def build_atmo2020(
    path: Path | str,
    *,
    filters: Sequence[Filter],
    filter_names: Sequence[str],
    progress_bar: bool = True,
) -> ModelGrid:
    """Build the ATMO2020 ModelGrid.

    Parameters
    ----------
    path : Path or str
        Path to directory containing grid_m1.0, grid_m0.5, grid_p0, grid_p0.3
        subdirectories with spectroscopy.
    filters : sequence of sedpy.observate.Filter
        Photometric filters.
    filter_names : sequence of str
        Names of photometric filters to store as metadata.
    progress_bar : bool, default=True
        Display a progress bar while loading the model.

    Returns
    -------
    model_grid : ModelGrid
        ATMO2020 ModelGrid object.

    Notes
    -----
    The ATMO2020 grid is complete.
    """

    def _parse_filename(name: str):
        """
        Parse Teff, logg, [M/H] from an ATMO2020 .dat filename.

        Returns log10(Teff), logg, mh
        """
        parts = name.split("_")
        teff = float(parts[2][1:])
        logg = float(parts[3][1:])
        mh_raw = parts[4]
        if mh_raw.startswith("m"):
            mh = -float(mh_raw[1:])
        elif mh_raw.startswith("p"):
            mh = float(mh_raw[1:])
        return np.log10(teff), logg, mh

    path = Path(path)
    values = []
    model_wave = None
    new_wave = LOWER_RES_WAVE
    for dir_name in ["grid_m1.0", "grid_m0.5", "grid_p0", "grid_p0.3"]:
        dir = path / dir_name
        glob = list(dir.glob("spec*"))
        for file_path in tqdm(glob, desc=dir_name) if progress_bar else glob:
            params = _parse_filename(file_path.name)
            data = pd.read_csv(
                file_path, sep=" ", comment="#", header=None, engine="c"
            ).values
            if model_wave is None:
                model_wave = data[:, 0] * 1e4  # Ang
                assert np.all(model_wave[1:] >= model_wave[:-1])
                new_wave = trim_wavelength_coverage(LOWER_RES_WAVE, model_wave)
            flux = data[:, 1] * 1e-1  # erg/s/cm^2/Ang
            arrays = (
                _filter_fluxes(model_wave, flux, filters) * 1e23 * 1e9,  # nJy
                flambda_to_fnu(
                    new_wave,  # Ang
                    resample_spectrum(new_wave, model_wave, flux),
                )
                * 1e23
                * 1e9,  # nJy
            )
            # Normalize so that scaling by R^2 / D^2 gives the correct value.
            # Currently normalized to R = 0.1 Rsun and D = 10 pc.
            # Scale by (10 pc / 0.1 Rsun)^2 = (100 pc / 1 Rsun)^2.
            arrays = (arrays[0] * 1.96724e19, arrays[1] * 1.96724e19)
            values.append((params, arrays))
    return _build_from_values(
        values,
        filter_names=filter_names,
        axes=["teff", "logg", "mh"],
        model_name="ATMO2020",
        wave=new_wave,
    )


# ============================================================
# Model: LOWZ
#
# Expected directory layout
# -------------------------
#   <model_path>/
#       LOWZ_models_index.csv  (col: TEFF LOGG METALLICITY CTOO LOGKZZ FILENAME)
#       models.tar.gz
#
# File format: whitespace-delimited ASCII, two columns:
#   col 0: wavelength in microns
#   col 1: flux in W/m^2/m  ->  multiply by 1e-7 to get erg/s/cm^2/Ang
# ============================================================
def build_lowz(
    path: Path | str,
    *,
    filters: Sequence[Filter],
    filter_names: Sequence[str],
    progress_bar: bool = True,
) -> ModelGrid:
    """Build the raw LOWZ ModelGrid.

    Parameters
    ----------
    path : Path or str
        Path to directory containing LOWZ_models_index.csv and models.tar.gz.
    filters : sequence of sedpy.observate.Filter
        Photometric filters.
    filter_names : sequence of str
        Names of photometric filters to store as metadata.
    progress_bar : bool, default=True
        Display a progress bar while loading the model.

    Returns
    -------
    model_grid : ModelGrid
        LOWZ ModelGrid object.
    """
    path = Path(path)
    index_path = path / "LOWZ_models_index.csv"
    models_path = path / "models.tar.gz"

    if not index_path.exists():
        raise ValueError(f"Error: index file not found: {index_path}")
    if not models_path.exists():
        raise ValueError(f"Error: models tarball not found: {models_path}")

    index = pd.read_csv(index_path).set_index("FILENAME")

    values = []
    model_wave = None
    new_wave = LOWER_RES_WAVE
    with (
        ProgressByteReader.open(models_path)
        if progress_bar
        else open(models_path, "rb")
    ) as fileobj:
        with tarfile.open(fileobj=fileobj, mode="r|gz") as in_tar:
            for member in in_tar:
                if not member.isfile():
                    continue
                input_file = in_tar.extractfile(member)
                if input_file is None:
                    continue
                file_bytes = input_file.read()
                stream = io.BytesIO(file_bytes)
                row = index.loc[Path(member.name).name]
                params = (
                    np.log10(row["TEFF"]),
                    row["LOGG"],
                    row["LOGKZZ"],
                    row["METALLICITY"],
                    row["CTOO"],
                )
                data = pd.read_csv(
                    stream, sep=" ", comment="#", header=None, engine="c"
                ).values
                if model_wave is None:
                    model_wave = data[:, 0] * 1e4
                    assert np.all(model_wave[1:] >= model_wave[:-1])
                    new_wave = trim_wavelength_coverage(
                        LOWER_RES_WAVE, model_wave
                    )
                flux = data[:, 1] * 1e-7  # erg/s/cm^2/Ang
                arrays = (
                    _filter_fluxes(model_wave, flux, filters)
                    * 1e23
                    * 1e9,  # nJy
                    flambda_to_fnu(
                        new_wave,  # Ang
                        resample_spectrum(new_wave, model_wave, flux),
                    )
                    * 1e23
                    * 1e9,  # nJy
                )
                values.append((params, arrays))
    return _build_from_values(
        values,
        filter_names=filter_names,
        axes=["teff", "logg", "kzz", "mh", "co"],
        model_name="LOWZ",
        wave=new_wave,
    )


def _build_from_values(
    values: Sequence[
        tuple[
            tuple[float, ...],
            tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]],
        ]
    ],
    *,
    filter_names: Sequence[str],
    axes: Sequence[str],
    model_name: str,
    wave: npt.NDArray[np.float32],
) -> ModelGrid:
    """Build a ModelGrid from a sequence of values.

    Each element of values is a tuple. The first element of the tuple is a tuple
    of float corresponding to the parameters. The second element of the tuple is
    a tuple of ndarray where the first element is the photometric flux values
    and the second element is the spectroscopic flux values.
    """
    points = tuple(
        np.unique(np.array(list(axis)))
        for axis in zip(*(params for params, _ in values))
    )
    base_shape = tuple(len(axis) for axis in points)
    grids = (
        np.full(base_shape + (len(filter_names),), np.nan, dtype=np.float32),
        np.full(base_shape + (len(wave),), np.nan, dtype=np.float32),
    )
    for params, arrays in values:
        ind = tuple(
            # Since the param must exist in the axis, searchsorted will return
            # the exact index of the param.
            np.searchsorted(axis, param)
            for param, axis in zip(params, points)
        )
        for grid, arr in zip(grids, arrays):
            grid[ind] = arr
    phot, spec = grids
    return ModelGrid(
        points=points,
        phot=phot,
        spec=spec,
        axes=tuple(axes),
        model_name=model_name,
        fill_method=None,
        filters=tuple(filter_names),
        wave=wave,
    )
