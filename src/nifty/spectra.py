"""Convert and resample spectral flux densities."""

import numpy as np
import numpy.typing as npt
from spectres import spectres
from spectres.spectral_resampling import make_bins

__all__ = [
    "flambda_to_fnu",
    "fnu_to_flambda",
    "resample_spectrum",
    "resample_reduction",
    "trim_wavelength_coverage",
    "trim_wavelength_coverage_mask",
]

_C_ANGSTROM_PER_SECOND = 2.99792458e18


def flambda_to_fnu(
    wave: npt.ArrayLike, flux: npt.ArrayLike
) -> npt.NDArray[np.floating]:
    """Convert F_lambda to F_nu.

    Parameters
    ----------
    wave : ndarray of shape (..., N)
        Wavelength in Angstroms. Broadcasts against flux.
    flux : ndarray of shape (..., N)
        F_lambda in spectral flux density per Angstrom.

    Returns
    -------
    flux_fnu : ndarray of shape (..., N)
        F_nu in spectral flux density per Hz.
    """
    wave = np.asarray(wave)
    flux = np.asarray(flux)
    return flux * np.square(wave) / _C_ANGSTROM_PER_SECOND


def fnu_to_flambda(
    wave: npt.ArrayLike, flux: npt.ArrayLike
) -> npt.NDArray[np.floating]:
    """Convert F_nu to F_lambda.

    Parameters
    ----------
    wave : ndarray of shape (..., N)
        Wavelength in Angstroms. Broadcasts against flux.
    flux : ndarray of shape (..., N)
        F_nu in spectral flux density per Hz.

    Returns
    -------
    flux_flambda : ndarray of shape (..., N)
        F_lambda in spectral flux density per Angstrom.
    """
    wave = np.asarray(wave)
    flux = np.asarray(flux)
    return flux * _C_ANGSTROM_PER_SECOND / np.square(wave)


def resample_spectrum(
    new_wave: npt.ArrayLike,
    wave: npt.ArrayLike,
    flux: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Resample F_lambda while conserving integrated flux.

    Parameters
    ----------
    new_wave : ndarray of shape (M,)
        Target wavelength grid in Angstroms.
    wave : ndarray of shape (N,)
        Source wavelength grid in Angstroms.
    flux : ndarray of shape (..., N)
        F_lambda in spectral flux density per Angstrom.

    Returns
    -------
    resampled : ndarray of shape (..., M)
        Resampled F_lambda in spectral flux density per Angstrom. Bins outside
        the source wavelength coverage contain NaN.

    Notes
    -----
    Both wavelength grids must have at least two strictly increasing samples.
    A target bin must fit entirely within the source bin edges to receive a
    resampled value.
    """
    return spectres(
        np.asarray(new_wave),
        np.asarray(wave),
        np.asarray(flux),
        fill=np.nan,
        verbose=False,
    )


def resample_reduction(
    wave_out: npt.ArrayLike, *, clip_to_coverage: bool = False
):
    """Create a chunk reducer for ModelGrid F_nu spectra.

    Parameters
    ----------
    wave_out : array_like of shape (M,)
        Target wavelengths in Angstroms.
    clip_to_coverage : bool, default=False
        Remove target bins that extend beyond the source wavelength coverage.

    Returns
    -------
    reduction : callable
        Called with the source wavelengths. Returns the target wavelengths and
        a callable that resamples flux chunks of shape (C, N) to (C, M).
        Input and output fluxes are F_nu in nJy.
    """
    target = np.asarray(wave_out, dtype=np.float32)

    def reduction(wave_in: npt.NDArray[np.float32]):
        selected = (
            target[trim_wavelength_coverage_mask(target, wave_in)]
            if clip_to_coverage
            else target
        )
        if selected.size < 2:
            raise ValueError(
                "Target wavelength grid has fewer than two covered bins."
            )

        def reduce_flux(
            flux: npt.NDArray[np.float32],
        ) -> npt.NDArray[np.float32]:
            flambda = fnu_to_flambda(wave_in, flux.astype(np.float64))
            resampled = resample_spectrum(selected, wave_in, flambda)
            return flambda_to_fnu(selected, resampled).astype(np.float32)

        return selected, reduce_flux

    return reduction


def trim_wavelength_coverage_mask(
    new_wave: npt.ArrayLike, wave: npt.ArrayLike
) -> npt.NDArray[np.bool_]:
    """Select target wavelength bins fully covered by a source wavelength grid.

    Both grids use bin edges halfway between adjacent wavelength samples.

    Parameters
    ----------
    new_wave : ndarray of shape (M,)
        Target wavelength grid.
    wave : ndarray of shape (N,)
        Source wavelength grid.

    Returns
    -------
    mask : ndarray of bool of shape (M,)
        True where the entire target bin is within the source bin edges.
    """
    wave = np.asarray(wave)
    new_wave = np.asarray(new_wave)
    edges, _ = make_bins(wave)
    new_edges, _ = make_bins(new_wave)
    return (new_edges[:-1] >= edges[0]) & (new_edges[1:] <= edges[-1])


def trim_wavelength_coverage(
    new_wave: npt.ArrayLike, wave: npt.ArrayLike
) -> npt.NDArray[np.floating]:
    """Trim a new wavelength grid based on the coverage of another grid.

    Parameters
    ----------
    new_wave : ndarray of shape (M,)
        Target wavelength grid.
    wave : ndarray of shape (N,)
        Source wavelength grid.

    Returns
    -------
    trimmed_wave : ndarray
        A copy of new_wave trimmed to the coverage of wave.
    """
    new_wave = np.asarray(new_wave)
    return new_wave[trim_wavelength_coverage_mask(new_wave, wave)]
