"""Convert and resample spectral flux densities."""

import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_matrix
from scipy.special import ndtr
from spectres import spectres
from spectres.spectral_resampling import make_bins

__all__ = [
    "convolve_lsf",
    "flambda_to_fnu",
    "fnu_to_flambda",
    "lsf_reduction",
    "resample_reduction",
    "resample_spectrum",
    "trim_wavelength_coverage",
    "trim_wavelength_coverage_mask",
]

_C_ANGSTROM_PER_SECOND = 2.99792458e18
_FWHM_TO_SIGMA = 1 / (2 * np.sqrt(2 * np.log(2)))


def _validate_lsf(
    wave: npt.NDArray[np.float64],
    lsf_wave: npt.NDArray[np.float64],
    dlds: npt.NDArray[np.float64],
    lsf_fwhm: float,
) -> None:
    if (
        wave.ndim != 1
        or wave.size < 2
        or not np.all(np.isfinite(wave))
        or not np.all(wave > 0)
        or not np.all(np.diff(wave) > 0)
    ):
        raise ValueError(
            "Wavelength grid must contain at least two finite, positive, increasing values."
        )
    if (
        lsf_wave.ndim != 1
        or dlds.ndim != 1
        or lsf_wave.size < 2
        or lsf_wave.shape != dlds.shape
        or not np.all(np.isfinite(lsf_wave))
        or not np.all(lsf_wave > 0)
        or not np.all(np.diff(lsf_wave) > 0)
        or not np.all(np.isfinite(dlds))
        or not np.all(dlds > 0)
    ):
        raise ValueError(
            "LSF wavelengths and dispersion must be finite and positive, with increasing wavelengths."
        )
    if not np.isfinite(lsf_fwhm) or lsf_fwhm <= 0:
        raise ValueError("LSF FWHM must be finite and positive.")


def _lsf_weights(
    wave: npt.NDArray[np.float64], sigma: npt.NDArray[np.float64]
) -> csr_matrix:
    """Integrate normalized Gaussian weights over source wavelength bins."""
    edges, _ = make_bins(wave)
    indices: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    for i, (center, width) in enumerate(zip(wave, sigma)):
        lower = center - 5 * width
        upper = center + 5 * width
        lo = max(0, int(np.searchsorted(edges, lower, side="right")) - 1)
        hi = min(len(wave), int(np.searchsorted(edges, upper, side="left")))
        left = np.maximum(edges[lo:hi], lower)
        right = np.minimum(edges[lo + 1 : hi + 1], upper)
        weights = ndtr((right - center) / width) - ndtr(
            (left - center) / width
        )
        weights /= weights.sum()
        indices.extend([i] * (hi - lo))
        columns.extend(range(lo, hi))
        values.extend(weights.tolist())
    return csr_matrix(
        (values, (indices, columns)), shape=(len(wave), len(wave))
    )


def convolve_lsf(
    wave: npt.ArrayLike,
    flux: npt.ArrayLike,
    lsf_wave: npt.ArrayLike,
    dlds: npt.ArrayLike,
    lsf_fwhm: float,
) -> npt.NDArray[np.float64]:
    """Convolve F_lambda with a wavelength-dependent Gaussian LSF.

    Parameters
    ----------
    wave : array_like of shape (N,)
        Strictly increasing model wavelengths in Angstroms.
    flux : array_like of shape (..., N)
        F_lambda in spectral flux density per Angstrom.
    lsf_wave : array_like of shape (L,)
        Strictly increasing LSF wavelengths in Angstroms. Must cover ``wave``.
    dlds : array_like of shape (L,)
        Positive dispersion in Angstroms per pixel at ``lsf_wave``.
    lsf_fwhm : float
        Gaussian full width at half maximum in pixels.

    Returns
    -------
    convolved : ndarray of shape (..., N)
        Convolved F_lambda on the input wavelength grid.

    Notes
    -----
    The dispersion is interpolated to ``wave``. Gaussian weights use source
    bin integrals, are truncated at five standard deviations, and are normalized
    at the grid boundaries.
    """
    wave = np.asarray(wave, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)
    lsf_wave = np.asarray(lsf_wave, dtype=np.float64)
    dlds = np.asarray(dlds, dtype=np.float64)
    _validate_lsf(wave, lsf_wave, dlds, lsf_fwhm)
    if flux.ndim < 1 or flux.shape[-1] != wave.size:
        raise ValueError("Flux must have one value per model wavelength.")
    if wave[0] < lsf_wave[0] or wave[-1] > lsf_wave[-1]:
        raise ValueError(
            "LSF file does not cover the required model convolution wavelengths."
        )
    sigma = np.interp(wave, lsf_wave, dlds) * lsf_fwhm * _FWHM_TO_SIGMA
    weights = _lsf_weights(wave, sigma)
    return np.asarray(flux @ weights.T)


def lsf_reduction(
    wave_out: npt.ArrayLike,
    lsf_wave: npt.ArrayLike,
    dlds: npt.ArrayLike,
    lsf_fwhm: float,
):
    """Create a chunk reducer that convolves and resamples ModelGrid spectra.

    Parameters
    ----------
    wave_out : array_like of shape (M,)
        Target wavelengths in Angstroms. Bins outside the model coverage are
        removed.
    lsf_wave : array_like of shape (L,)
        Strictly increasing LSF wavelengths in Angstroms.
    dlds : array_like of shape (L,)
        Positive dispersion in Angstroms per pixel at ``lsf_wave``.
    lsf_fwhm : float
        Gaussian full width at half maximum in pixels.

    Returns
    -------
    reduction : callable
        Called once with the source wavelengths. Returns the selected target
        wavelengths and a callable mapping F_nu chunks of shape (C, N) to
        convolved, resampled F_nu chunks of shape (C, M_selected), in nJy.

    Notes
    -----
    The source grid is trimmed to the target bins plus five standard deviations
    of LSF support. The LSF must cover the fitted bins, while the model must
    cover the Gaussian support of every convolution center. Outside the LSF
    table, dispersion is held at the nearest tabulated value. Gaussian weights are
    built once when the source wavelengths are supplied, then reused for every
    flux chunk. Convolution is performed in F_lambda before flux-conserving
    resampling and conversion back to F_nu.
    """
    target = np.asarray(wave_out, dtype=np.float32)
    lsf_wave = np.asarray(lsf_wave, dtype=np.float64)
    dlds = np.asarray(dlds, dtype=np.float64)

    def reduction(wave_in: npt.NDArray[np.float32]):
        source = np.asarray(wave_in, dtype=np.float64)
        _validate_lsf(source, lsf_wave, dlds, lsf_fwhm)
        if (
            target.ndim != 1
            or target.size < 2
            or not np.all(np.isfinite(target))
            or not np.all(target > 0)
            or not np.all(np.diff(target) > 0)
        ):
            raise ValueError(
                "Target wavelengths must contain at least two finite, positive, increasing values."
            )
        target_edges, _ = make_bins(target)
        source_edges, _ = make_bins(source)
        sigma = np.interp(source, lsf_wave, dlds) * lsf_fwhm * _FWHM_TO_SIGMA
        supported = np.zeros(target.size, dtype=bool)
        for i in range(target.size):
            left, right = target_edges[i : i + 2]
            if left < max(source_edges[0], lsf_wave[0]) or right > min(
                source_edges[-1], lsf_wave[-1]
            ):
                continue
            # Source bins overlapping this target bin need convolved values.
            first_center = int(
                np.searchsorted(source_edges[1:], left, side="right")
            )
            last_center = int(
                np.searchsorted(source_edges[:-1], right, side="left")
            )
            centers = source[first_center:last_center]
            widths = sigma[first_center:last_center]
            supported[i] = (
                centers.size > 0
                and np.all(centers - 5 * widths >= source_edges[0])
                and np.all(centers + 5 * widths <= source_edges[-1])
            )
        selected = target[supported]
        if selected.size < 2:
            raise ValueError(
                "Target wavelength grid has fewer than two bins covered by the LSF and model convolution support."
            )
        selected_indices = np.flatnonzero(supported)
        left = float(target_edges[selected_indices[0]])
        right = float(target_edges[selected_indices[-1] + 1])
        in_bins = (source_edges[:-1] < right) & (source_edges[1:] > left)
        padding = 5 * float(np.max(sigma[in_bins]))
        lo = left - padding
        hi = right + padding
        # Include the neighboring model samples so target bins remain covered.
        first = max(0, int(np.searchsorted(source, lo)) - 1)
        last = min(
            len(source), int(np.searchsorted(source, hi, side="right")) + 1
        )
        trimmed = source[first:last]
        sigma = np.interp(trimmed, lsf_wave, dlds) * lsf_fwhm * _FWHM_TO_SIGMA
        weights = _lsf_weights(trimmed, sigma)

        def reduce_flux(
            flux: npt.NDArray[np.float32],
        ) -> npt.NDArray[np.float32]:
            flambda = fnu_to_flambda(
                trimmed, flux[:, first:last].astype(np.float64)
            )
            convolved = np.asarray(flambda @ weights.T)
            # Resample on the original grid before selecting bins: recomputing
            # edges from a filtered grid would change neighboring bin widths.
            resampled = resample_spectrum(target, trimmed, convolved)[
                :, supported
            ]
            return flambda_to_fnu(selected, resampled).astype(np.float32)

        return selected, reduce_flux

    return reduction


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
