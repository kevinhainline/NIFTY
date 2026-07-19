"""Forward modeling parameters."""

import io
import json
import tarfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

import numpy as np
import numpy.typing as npt
from scipy.interpolate import RegularGridInterpolator, interp1d
from tqdm import tqdm

__all__ = ["IndexMap", "Model", "ModelGrid", "linear_teff"]


class IndexMap(Mapping):
    """
    Immutable mapping of keys to unique indices.

    Parameters
    ----------
    keys : iterable of str
        The keys to use for the index map.
    """

    def __init__(self, keys: Iterable[str]):
        self._map = {k: i for i, k in enumerate(dict.fromkeys(keys))}

    def __getitem__(self, key):
        return self._map[key]

    def __iter__(self):
        return iter(self._map)

    def __len__(self):
        return len(self._map)


@dataclass(frozen=True)
class ModelGrid:
    """
    ModelGrid for estimating photometry or spectroscopy from parameters.

    Parameters
    ----------
    points : tuple of ndarray of float of shapes (d1, ), ..., (dN, )
        Grid points on each axis in N dimensions.
    phot : ndarray of shape (d1, ..., dN, F)
        Photometry data grid in units nJy. F is the number of filters.
    spec : ndarray of shape (d1, ..., dN, W)
        Spectroscopy data grid in f_nu in units nJy. W is the number of points
        on the wave.
    axes : tuple of str
        Axis names in same order as points.
    model_name : str
        Name of model used to estimate values.
    fill_method : str
        How missing grid points are filled or None if unfilled.
    filters : tuple of str
        The filter names used for photometry.
    wave : ndarray of shape (W,)
        The sampled points on the wave used for spectroscopy in Angstroms.
    """

    points: tuple[np.ndarray[tuple[int], np.dtype[np.float32]], ...]
    phot: npt.NDArray[np.float32]
    spec: npt.NDArray[np.float32]
    axes: tuple[str, ...]
    model_name: str
    fill_method: str | None
    filters: tuple[str, ...]
    wave: np.ndarray[tuple[int], np.dtype[np.float32]]

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Load a ModelGrid from a file.

        Parameters
        ----------
        path : str
            Path to the file to load.

        Returns
        -------
        obj : ModelGrid
            An instance of the ModelGrid loaded from the file.
        """
        with tarfile.open(path, mode="r|gz") as tar:
            meta_member = tar.next()
            if meta_member is None:
                raise ValueError("Tarfile is empty.")
            meta_file = tar.extractfile(meta_member)
            assert meta_file is not None
            meta = json.loads(meta_file.read().decode("utf-8"))
            arrays = {}
            while True:
                member = tar.next()
                if member is None:
                    break
                name = Path(member.name).stem
                array_file = tar.extractfile(member)
                assert array_file is not None
                arrays[name] = np.load(io.BytesIO(array_file.read()))
            points = tuple(arrays[name] for name in meta["axes"])
        return cls(
            points=points,
            phot=arrays["phot"],
            spec=arrays["spec"],
            axes=tuple(meta["axes"]),
            model_name=meta["model_name"],
            fill_method=meta["fill_method"],
            filters=tuple(
                f.lower() if "_" in f else "jwst_" + f.lower()
                for f in meta["filters"]
            ),
            wave=arrays["wave"],
        )

    def save(self, path: str | Path) -> None:
        """
        Save a ModelGrid to a file.

        Parameters
        ----------
        path : str
            The path to save the ModelGrid.
        """

        # Write JSON metadata to bytes.
        def _write_meta(meta):
            buffer = io.BytesIO(json.dumps(meta).encode("utf-8"))
            return buffer.getvalue()

        # Write an ndarray to bytes.
        def _write_array(obj):
            buffer = io.BytesIO()
            np.save(buffer, obj)
            return buffer.getvalue()

        # Add some bytes to a tar file.
        def _add_bytes_to_tar(bytes, name, tar):
            stream = io.BytesIO(bytes)
            tar_info = tarfile.TarInfo(name=name)
            tar_info.size = len(bytes)
            tar.addfile(tar_info, fileobj=stream)

        meta = {}
        meta["model_name"] = self.model_name
        meta["fill_method"] = self.fill_method
        meta["filters"] = self.filters
        meta["axes"] = self.axes

        arrays = {}
        arrays["phot"] = self.phot
        arrays["spec"] = self.spec
        arrays["wave"] = self.wave
        for name, axis in zip(meta["axes"], self.points):
            arrays[name] = axis
        with tarfile.open(path, mode="w|gz") as tar:
            _add_bytes_to_tar(_write_meta(meta), "meta.json", tar)
            for name, array in arrays.items():
                _add_bytes_to_tar(_write_array(array), f"{name}.npy", tar)

    def with_filters(self, filters: tuple[str, ...]) -> Self:
        """A new ModelGrid object with photometry using a subset of filters.

        Parameters
        ----------
        filters : tuple of str
            Filters used to resample photometry. All filters must already be
            present in the ModelGrid. Otherwise, an error is raised.

        Returns
        -------
        resampled_model_grid : ModelGrid
            The model_grid resampled to new filters or a new wave.

        Notes
        -----
        The filters are resampled by extracting only the specified filters from
        the filters in the ModelGrid. Therefore, if a filter does not exist in
        the ModelGrid, an error is raised. The ModelGrid needs to be recreated
        with additional filters.
        """
        return self.__class__(
            points=self.points,
            phot=_change_filters(filters, self.filters, self.phot),
            spec=self.spec,
            axes=self.axes,
            model_name=self.model_name,
            fill_method=self.fill_method,
            filters=filters,
            wave=self.wave,
        )

    def with_wave(self, wave: npt.NDArray[np.float32]):
        """A new ModelGrid object with spectroscopy using a difference wave.

        Parameters
        ----------
        wave : ndarray
            Wave to resample ModelGrid.

        Returns
        -------
        resampled_model_grid : ModelGrid
            The model_grid resampled to new filters or a new wave.

        Notes
        -----
        The wave is resampled by linear interpolation. The difference in flux
        conservation is negligible compared to the error of grid interpolation.
        """
        return ModelGrid(
            points=self.points,
            phot=self.phot,
            spec=_resample_spectrum(wave, self.wave, self.spec),
            axes=self.axes,
            model_name=self.model_name,
            fill_method=self.fill_method,
            filters=self.filters,
            wave=wave,
        )


@dataclass(frozen=True)
class Model:
    """
    Forward model parameters to data.

    Parameters
    ----------
    interp : RegularGridInterpolator
        Interpolator object to use for modeling.
    axes_map : IndexMap
        Index map for axes. Must have a distance axis with key 'd'.

    Attributes
    ----------
    n_params : int
        Number of parameters in the model.
    n_interp_params : int
        Number of interpolated parameters in the model.
    """

    interp: RegularGridInterpolator
    axes_map: IndexMap
    n_params: int = field(init=False)
    n_interp_params: int = field(init=False)
    _d_index: int = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "n_params", len(self.axes_map))
        object.__setattr__(self, "n_interp_params", len(self.interp.grid))
        object.__setattr__(self, "_d_index", self.axes_map["d"])

    @classmethod
    def phot(
        cls,
        model_grid: ModelGrid,
        filters: Sequence[str] | None = None,
    ) -> Self:
        """
        Create a Model from the photometry data in a ModelGrid.

        The Model can also use a new set of filters that are a subset of those
        in the ModelGrid.

        Parameters
        ----------
        model_grid : ModelGrid
            ModelGrid used to create the interpolator.
        filters : tuple of str, optional
            Tuple of filter names to use from model_grid. If not provided, the
            filters are not changed from what is in model_grid.

        Returns
        -------
        obj : Model
            Model from photometry data.
        """
        if filters is not None:
            values = _change_filters(
                filters, model_grid.filters, model_grid.phot
            )
        else:
            values = model_grid.phot
        interp = RegularGridInterpolator(
            model_grid.points,
            values,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )
        axes_map = IndexMap(model_grid.axes + ("d",))
        return cls(interp, axes_map)

    @classmethod
    def spec(
        cls,
        model_grid: ModelGrid,
        wave: npt.NDArray[np.float32] | None = None,
    ) -> Self:
        """
        Create a Model from the spectroscopy data in a ModelGrid.

        The Model can also use a new wave.

        Parameters
        ----------
        model_grid : ModelGrid
            ModelGrid used to create the interpolator.
        wave : ndarray, optional
            New wave to resample model_grid data. If not provided, the wave is
            not changed from what is in the model_grid.

        Returns
        -------
        obj : Model
            Model from spectroscopy data.
        """
        if wave is not None:
            values = _resample_spectrum(wave, model_grid.wave, model_grid.spec)
        else:
            values = model_grid.spec
        interp = RegularGridInterpolator(
            model_grid.points,
            values,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )
        axes_map = IndexMap(model_grid.axes + ("d",))
        return cls(interp, axes_map)

    def __call__(
        self, theta: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        """
        Estimate values for parameters.

        Parameters
        ----------
        theta : ndarray of shape (..., N)
            Parameters at which to estimate. The parameters must match the
            axes_map of this Model.

        Returns
        -------
        model_flux : ndarray of shape (..., F)
            Model flux in nJy. The shape is the same as theta except the last
            axis is the flux.
        """
        model_flux = self.interp(theta[..., : self.n_interp_params])  # nJy
        object_radius = 0.10276 * 2.2555823856078e-8  # pc
        model_flux *= np.square(object_radius / theta[..., self._d_index])[
            ..., np.newaxis
        ]
        return model_flux


def _resample_spectrum(
    new_wave: npt.NDArray[np.float32],
    wave: npt.NDArray[np.float32],
    flux: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """
    Resample the spectrum of a flux data grid.

    Uses the last axis of flux as the spectrum.
    """
    interp = interp1d(wave, flux, axis=-1, fill_value="extrapolate")
    return interp(new_wave)


def _change_filters(
    new_filters: Sequence[str],
    filters: Sequence[str],
    flux: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """
    Select a subset of filters from a flux data grid.

    Raises KeyError if some filters in new_filters are not present in filters.
    """
    name_to_idx = {name: i for i, name in enumerate(filters)}
    idx = [name_to_idx[name] for name in new_filters]
    return np.take(flux, idx, axis=-1)


def fill_point(
    points: tuple[npt.NDArray[np.float32], ...],
    values: npt.NDArray[np.float32],
    target: tuple[int, ...],
) -> npt.NDArray[np.float32]:
    """
    Estimate the value of a missing point in a grid by fitting a polynomial.

    Parameters
    ----------
    points : tuple of ndarray of float, of shapes (d1, ), ..., (dN, )
        The points defining the regular grid in N dimensions.
    values : array_like, shape (d1, ..., dN, M)
        The data on the regular grid in N dimensions. Each value on the grid
        is a vector of length M.
    target: tuple of N ints
        The index of the target point in N dimensions.

    Returns
    --------
    value : ndarray of shape (M,)
        The estimated data vector at the target.
    """
    # The number of parameters.
    N = len(points)

    # The length of the output vector.
    M = values.shape[-1]

    # Slice the hypercube.
    cube_starts = []
    cube_points = []
    for coord, axis_points, axis_len in zip(target, points, values.shape):
        if axis_len < 3:
            raise ValueError(
                "Unable to construct hypercube: "
                "all axes must have a length of at least 3. "
                f"Recieved axis lengths {values.shape[:-1]}."
            )
        start = coord - 1
        # Shift the hypercube if it goes out of bounds.
        start = max(start, 0)
        start = min(start, axis_len - 3)
        cube_starts.append(start)

        # Extract the physical values for this axis window.
        cube_points.append(axis_points[start : start + 3])

    # Get the slices for extracting the hypercube.
    slices = tuple(slice(start, start + 3) for start in cube_starts)
    # Extract the entire output vector.
    slices = slices + (slice(None),)

    # Y is an array of shape (n_points, M) where
    # Y[i, :] represents the output vector at point i.
    Y = values[slices].reshape(-1, M)

    # axis_points is an array of shape (N, 3) where
    # axis_points[i, j] is coordinate j on axis i.
    # The cube edge has three points, so 0 <= j < 3.
    axis_points = np.stack(cube_points)  # shape (N, 3)

    target_rel_cube = tuple(t - s for t, s in zip(target, cube_starts))
    target_point = axis_points[np.arange(N), target_rel_cube]

    # Get coordinates that are not the same as the target coordinate.
    rem_mask = np.asarray(target_rel_cube)[:, np.newaxis] != np.arange(3)
    remaining = axis_points[rem_mask].reshape(-1, 2)
    total_dist = np.abs(target_point - remaining[:, 1]) + np.abs(
        target_point - remaining[:, 0]
    )

    # Arrays of shape (N,) denoting properties for each axis.
    axis_space = axis_points[:, 2] - axis_points[:, 0]
    axis_center = axis_points[:, 1]

    # X is an array of shape (n_points, N) where
    # X[i, :] represents the physical coordinates at point i.
    mesh_grids = np.meshgrid(*cube_points, indexing="ij")
    X = np.stack(mesh_grids, axis=-1).reshape(-1, N)

    # Identify points containing any NaN element in their output vector.
    mask = np.isnan(Y).any(axis=-1)

    # Mask the target point.
    # This is not necessary if the target point's value is known to be NaN.
    # We only set this for testing the fit at existing grid points.
    mask[np.ravel_multi_index(target_rel_cube, (3,) * N)] = True

    X = X[~mask]
    Y = Y[~mask]

    # Define the weight vector of shape (n_points,).
    # The weight along one axis is 1 - |x - t| / d, where x is the point
    # being weighted, t is the target point, and d is the total distance
    # between the target point and each of the other two points. The weight
    # of a point is the product of its weight along all axes.

    w = np.prod(1 - np.abs((X - target_point) / total_dist), axis=1)
    # w = np.prod(1 - np.square((X - target_point) / total_dist), axis=1)
    # w = np.full((X.shape[0],), 1.0, dtype=values.dtype)

    # Normalize the input vectors for numerical stability.
    X = (X - axis_center) / axis_space

    # We need at least 2^N points to solve a multilinear polynomial.
    num_terms = 2**N
    if len(X) < num_terms:
        raise ValueError(
            f"Insufficient valid points ({len(X)}) to fit a "
            f"multilinear polynomial requiring {num_terms} coefficients."
        )

    # Construct the design matrix for a multilinear polynomial.
    # For N variables, we require 2^N terms (combinations of subsets).

    # Assign every N-bit integer to a term.
    # term_i has shape (2^N,).
    term_i = np.arange(num_terms, dtype=np.int32)

    # Determine which variables should be included in each term.
    # include_term[i, j] is True when term i contains variable j.
    # include_term has shape (2^N, N).
    include_term = (term_i[:, np.newaxis] >> np.arange(N)) & 1

    # If the bit is 0, we want 1 (so it doesn't affect the product).
    # If the bit is 1, we want the variable value from X.
    # X[:, np.newaxis, :] has shape (n_points, 1, N).
    # include_term[np.newaxis, ...] has shape (1, 2^N, N).
    # factors has shape (n_points, 2^N, N).
    factors = np.where(include_term[np.newaxis, ...], X[:, np.newaxis, :], 1.0)

    # Multiply along the variable axis (axis=2) to get shape (n_points, 2^N).
    A = np.prod(factors, axis=2)

    # To perform, weighted least squares, we modify A and Y.
    # If W is the matrix with w as its diagonal, this is the equivalent of
    # A' = W^(1/2)A and Y' = W^(1/2)Y.
    # Solving with these new values using OLS is equivalent to WLS.
    w_sqrt = np.sqrt(w)[:, np.newaxis]
    A *= w_sqrt
    Y *= w_sqrt

    C, _, _, _ = np.linalg.lstsq(A, Y, rcond=None)

    # Solve the polynomial at the target point.
    # See above for how the design matrix is created.
    # This is similar, except that only one row is needed for the one point.
    target_norm = (target_point - axis_center) / axis_space
    factors = np.where(include_term, target_norm[np.newaxis, :], 1.0)
    a_target = np.prod(factors, axis=1)

    # Compute the normalized predictions: shape (1, M).
    y_target = a_target @ C

    return y_target


def fill_model_fit(
    model_grid: ModelGrid, progress_bar: bool = False
) -> ModelGrid:
    """
    Return a new model with missing points filled by a fitted polynomial.

    Parameters
    ----------
    model_grid : ModelGrid
        Model grid to fill in place.
    progress_bar : bool , default=False
        Whether to show a progress bar for filling.
    """
    grids = []
    for name, grid in [("phot", model_grid.phot), ("spec", model_grid.spec)]:
        # Get all indices that are missing points (NaN).
        # Grid has shape (d1, ..., dN, M)
        # nan_mask has shape (d1, ..., dN)
        nan_mask = np.isnan(grid).any(axis=-1)
        # missing has shape (number_missing, N)
        missing = np.argwhere(nan_mask)
        missing_list = [tuple(index) for index in missing]
        new_grid = grid.copy()
        for target in (
            tqdm(missing_list, desc=name) if progress_bar else missing_list
        ):
            new_grid[target] = fill_point(model_grid.points, grid, target)
        grids.append(new_grid)
    phot, spec = grids
    return ModelGrid(
        points=model_grid.points,
        phot=phot,
        spec=spec,
        axes=model_grid.axes,
        model_name=model_grid.model_name,
        fill_method="fit",
        filters=model_grid.filters,
        wave=model_grid.wave,
    )


def linear_teff(
    a: npt.NDArray[np.float32], axes_map: IndexMap, inplace=False
) -> npt.NDArray[np.float32]:
    """Convert effective temperature from logarithmic to linear space.

    Parameters
    ----------
    a : ndarray of shape (..., P)
        Array with logarithmic effective temperature values to convert.
        Parameters are along the last axis.
    axes_map : IndexMap
        Index mapping followed by the last axis of the input array.
    inplace : bool, default=False
        If True, the array will be modified in-place.

    Returns
    -------
    out : ndarray of shape (..., P)
        Array with logarithmic effective temperature values converted to linear.
    """
    if inplace:
        out = a
    else:
        out = a.copy()
    i = axes_map["teff"]
    out[..., i] = 10 ** out[..., i]
    return out
