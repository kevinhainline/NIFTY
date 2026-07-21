"""Main NIFTY CLI."""

import argparse
import time
import warnings
from functools import partial
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sedpy.observate import Filter

from ..load import (
    load_filter_info,
    load_phot_catalog_fits,
    load_phot_catalog_text,
    load_spec_text,
)
from ..model import Model, ModelGrid, linear_teff
from ..plot import plot_corner, plot_photometry, plot_spectrum
from ..prob import BayesianProbability
from ..sample import sample_mcmc, sample_nautilus
from .print import (
    print_banner,
    print_filters,
    print_id,
    print_model_grid_range,
    print_separator,
)

# Uses system's external TeX engine instead of matplotlib's.
matplotlib.rcParams["text.usetex"] = True
warnings.filterwarnings("ignore")


def _min_rel_error(flux, error, min_rel_error=0.05):
    """Restrict error to a minimum relative error and return changed indices.

    If the error is negative, it is ignored since this indicates bad data. The
    changed indices are used for warnings.

    Parameters
    ----------
    flux : ndarray of shape (N,)
        Object flux.
    error : ndarray of shape (N,)
        Object error.
    min_rel_error : float, default=0.05
        Minimum error relative to flux.

    Returns
    -------
    clipped_error : ndarray of shape (N,)
        Error with values below the minimum error set to the minimum error.
    changed : ndarray
        Indices of changed elements.
    """
    min_error = min_rel_error * np.abs(flux)
    clipped_error = np.maximum(error, min_error)
    # This includes NaN, which is falsy and negative errors.
    # In those cases, we set the error to its original value.
    clipped_error = np.where(error < 0, error, clipped_error)
    changed = np.nonzero((error < min_error) & (error >= 0))[0]
    return clipped_error, changed


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NIFTY: Near-Infrared Fitting for T and Y Dwarfs."
    )

    # Use a different parser for phot vs spec mode.
    subparsers = parser.add_subparsers(
        help="Fitting mode.",
        dest="mode",
        required=True,
    )

    # Arguments shared by both phot and spec mode.
    # This parser does not need help because it is handled by the parsers that
    # use it (phot and spec).
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        "-m",
        "--model",
        help="Path to model grid file used for forward modeling.",
        required=True,
    )
    parent_parser.add_argument(
        "-o",
        "--output",
        help="Path to output files. Default is '<model_name>_output/'.",
    )
    parent_parser.add_argument(
        "-S",
        "--stub",
        help="Survey stub to prepend to output filenames (e.g. JADES-GS).",
        required=True,
    )
    parent_parser.add_argument(
        "-s",
        "--sampler",
        help="Sampler to estimate posterior.",
        choices=["mcmc", "nautilus"],
        default="mcmc",
    )

    # Parser for photometry mode.
    phot_parser = subparsers.add_parser(
        "phot", help="Fit photometry.", parents=[parent_parser]
    )
    phot_parser.add_argument("catalog", help="Photometry catalog file.")
    phot_parser.add_argument(
        "--config", help="Filter configuration file.", required=True
    )
    phot_parser.add_argument(
        "-f",
        "--format",
        choices=["text", "fits"],
        help=(
            "Catalog file format. If not provided, guessed from the filename. "
            "'text' requires a space-delimited table containing flux and error "
            "columns that match the filter configuration. "
            "'fits' requires named HDU extensions and columns that match the "
            "filter configuration."
        ),
    )
    id_group = phot_parser.add_mutually_exclusive_group(required=True)
    # action="append" means that specifying this option multiple times will
    # append the passed argument to a list. For example, "-i 0 -i 1" -> [0, 1].
    id_group.add_argument(
        "-i",
        "--id",
        help=(
            "Object ID number to fit. Use this flag multiple times once per "
            "object ID to fit multiple objects."
        ),
        type=int,
        action="append",
        dest="obj_id",
    )
    id_group.add_argument(
        "--idlist",
        help=(
            "Path to file containing whitespace delimited list of object ID "
            "number(s) to fit."
        ),
    )
    phot_parser.add_argument(
        "--frac_model_floor", help="Fractional model error floor.", type=float
    )

    # Parser for spectroscopy mode.
    spec_parser = subparsers.add_parser(
        "spec", help="Fit spectroscopy.", parents=[parent_parser]
    )
    spec_parser.add_argument(
        "spectrum",
        help=(
            "Spectrum file. Requires a whitespace-delimited text file with "
            "columns for wave, flux, and error."
        ),
    )
    spec_parser.add_argument(
        "--id",
        help="Object ID number.",
        type=int,
        required=True,
        dest="obj_id",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()
    print_banner()

    # Load the model grid.
    print(f"Loading the model grid from {args.model}")
    model_grid = ModelGrid.load(args.model)
    print(f"Loaded model grid for {model_grid.model_name}.")
    print("  The model parameter range explored:")
    print_model_grid_range(model_grid, indent=3)
    print_separator()

    # Create the output directory.
    if args.output is not None:
        out_dir = Path(args.output)
    else:
        out_dir = Path(f"{model_grid.model_name}_output")
    if out_dir.exists() and not out_dir.is_dir():
        # The path exists, but it is not a directory, so we need to delete it.
        out_dir.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load the object data and create the model with the correct filters / wave.
    if args.mode == "phot":
        print(f"Opening config/filters json file: {args.config}")
        filter_info = load_filter_info(args.config)
        filter_names = tuple(str(info["name"]) for info in filter_info)
        filter_central_wavelength = np.array(
            [Filter(name).wave_pivot / 1e4 for name in filter_names],
            dtype=np.float32,
        )  # um
        print_filters(filter_names)

        # Create the model with an extra distance parameter.
        model = Model.phot(model_grid, filter_names)

        # Load the ID list file or use the ID arguments.
        if args.idlist is not None:
            with open(args.idlist, "r") as f:
                lines = f.readlines()
            ids = [int(line) for line in lines]
        else:
            ids = args.obj_id

        if args.format is not None:
            format = args.format
        else:
            suffixes = Path(args.catalog).suffixes
            # Use the last suffix
            suffix = suffixes[-1]
            if suffix == ".gz" and len(suffixes) > 1:
                # If this is compressed FITS, use the second to last suffix.
                suffix = suffixes[-2]
            if suffix in [".fits", ".fit", ".fts"]:
                format = "fits"
            else:
                format = "text"
        if format == "fits":
            flux, error = load_phot_catalog_fits(args.catalog, filter_info, ids)
        else:
            try:
                flux, error = load_phot_catalog_text(
                    args.catalog, filter_info, ids
                )
            except Exception as e:
                raise ValueError(
                    f"Could not parse text file '{args.catalog}'. "
                    "Make sure it's whitespace-delimited.\n"
                    f"See README for expected format.\nOriginal error: {e}"
                )
        objects = [obj for obj in zip(ids, flux, error)]
    elif args.mode == "spec":
        wave, flux, error = load_spec_text(args.spectrum)
        model = Model.spec(model_grid, wave)
        objects = [(args.obj_id, flux, error)]

    # We have to create a new model for spectroscopy, since NIFTY uses the
    # original model grid's wavelength for its SED.
    spec_model = Model.spec(model_grid)

    # Set up the probability object and the sampler.
    if args.mode == "spec" or args.frac_model_floor is None:
        frac_model_floor = 0.0
    else:
        frac_model_floor = args.frac_model_floor
    prob = BayesianProbability(model, frac_model_floor=frac_model_floor)
    # Pick the sampler function and assign keyword arguments.
    if args.sampler == "mcmc":
        sampler = partial(sample_mcmc, n_walkers=100, n_steps=100000)
    else:
        sampler = partial(sample_nautilus, n_live=2000)

    # List of tuple: (obj_id, time)
    convergence_times = []

    # Loop over all objects.
    for i, (obj_id, obs_flux, obs_error) in enumerate(objects):
        file_prefix = f"{args.stub}_{obj_id:06d}_{model_grid.model_name}"

        # Get the filepath for this object given a name (e.g. "SED.txt")
        def get_filepath(name: str) -> Path:
            return out_dir / f"{file_prefix}_{name}"

        print(f"\n\nFitting Object {i + 1} / {len(objects)}\n")
        print_id(obj_id, model_grid.model_name)
        print()
        if args.mode == "phot":
            # Limit obs_error to the minimum relative error and print filters
            # that changed.
            obs_error, changed = _min_rel_error(
                obs_flux, obs_error, min_rel_error=0.05
            )
            for changed_index in changed:
                print(
                    "     Updating flux error in "
                    f"{filter_names[changed_index]} to minimum relative error."
                )
            print_separator(indent=4)
        print("    Assuming source is at 1 Jupiter radius.")
        print_separator(indent=4)

        # Run sampler.
        print("    Initializing and running the sampler.")
        backend_file = get_filepath("sampler_backend.h5")
        time_start = time.time()
        samples = sampler(
            prob,
            obs_flux,
            obs_error,
            backend_file=backend_file,
        )
        time_end = time.time()
        convergence_time = (time_end - time_start) / 60  # minutes
        convergence_times.append((obj_id, convergence_time))
        print(
            f"    {args.stub}-{obj_id:06d}, fitting took "
            f"{convergence_time:.1f} minutes to converge"
        )
        print_separator(indent=4)

        # Plot corner.
        print("    Plotting corner plot")
        fig = plt.figure(figsize=(9, 9))
        plot_corner(fig, samples, axes_map=model.axes_map)
        fig.savefig(get_filepath("corner.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)
        print_separator(indent=4)

        # Build model percentile envelopes.
        print("    Getting median/upper/lower limits from the spectrum")
        spec_model_lower, spec_model_median, spec_model_upper = (
            samples.model_quantile(spec_model, [0.16, 0.5, 0.84])
        )
        model_median = samples.model_quantile(model, 0.5)
        print_separator(indent=4)

        # Compute chi-square reduced.
        valid_flux = ~np.isnan(obs_flux) & (obs_error > 0)
        chi_sq_model = np.sum(
            np.square(
                (obs_flux[valid_flux] - model_median[valid_flux])
                / obs_error[valid_flux]
            )
        )
        num_data_points = np.count_nonzero(valid_flux)
        num_free_params = model.n_params
        degrees_of_freedom = num_data_points - num_free_params
        if degrees_of_freedom > 0:
            chi_sq_red_model = chi_sq_model / degrees_of_freedom
        else:
            chi_sq_red_model = -9999
            print(
                "      Warning: number of free parameters >= number of data "
                "points, reduced chi-square will be -9999"
            )

        print("    Plotting SED / spectrum")
        fig = plt.figure(figsize=(8, 4))
        ax = fig.add_subplot(1, 1, 1)
        if args.mode == "phot":
            plot_photometry(
                ax,
                central_wave=filter_central_wavelength,
                obj_flux=obs_flux,
                obj_error=obs_error,
                obj_id=obj_id,
                model_phot_flux=model_median,
                model_spec_wave=model_grid.wave,
                model_spec_flux=spec_model_median,
                model_spec_flux_lower=spec_model_lower,
                model_spec_flux_upper=spec_model_upper,
                chi_sq_red=chi_sq_red_model,
            )
        elif args.mode == "spec":
            plot_spectrum(
                ax,
                wave=wave,
                obj_flux=obs_flux,
                obj_error=obs_error,
                obj_id=obj_id,
                model_wave=model_grid.wave,
                model_flux=spec_model_median,
                model_flux_lower=spec_model_lower,
                model_flux_upper=spec_model_upper,
                chi_sq_red=chi_sq_red_model,
            )
        fig.savefig(get_filepath("SED.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)
        print_separator(indent=4)

        print("    Creating Output Files ")

        # Write model estimated parameters to a file.
        param_lower, param_median, param_upper = linear_teff(
            samples.quantile([0.16, 0.5, 0.84]), model.axes_map, inplace=True
        )
        header = (
            "# ID"
            + "".join(
                [
                    f" {name}_lower {name} {name}_upper"
                    for name in model.axes_map.keys()
                ]
            )
            + " chisq chisq_reduced\n"
        )
        parameter_line = (
            str(obj_id)
            + "".join(
                [
                    f" {lower:.2f} {median:.2f} {upper:.2f}"
                    for lower, median, upper in zip(
                        param_lower, param_median, param_upper
                    )
                ]
            )
            + f" {chi_sq_model:.2f} {chi_sq_red_model:.2f}"
        )
        with open(get_filepath("parameters.txt"), "w") as f:
            f.write(header)
            f.write(parameter_line)

        # Write SED file (original model grid spectrum wavelength).
        np.savetxt(
            get_filepath("SED.txt"),
            np.column_stack(
                (
                    model_grid.wave / 1e4,
                    spec_model_lower,
                    spec_model_median,
                    spec_model_upper,
                )
            ),
            fmt="%f %f %f %f",
            header="Wavelength_um Flux_l68 Flux_50 Flux_u68",
        )

        # Write the model output that NIFTY fitted to the data.
        np.savetxt(
            get_filepath(
                "model_photometry.txt"
                if args.mode == "phot"
                else "model_spectroscopy.txt"
            ),
            np.column_stack(
                (
                    (
                        filter_central_wavelength
                        if args.mode == "phot"
                        else wave / 1e4
                    ),
                    model_median,
                )
            ),
            fmt="%f %f",
            header="Wavelength_um Flux_Model",
        )
        print_separator(indent=4)

        print("    Output files:")
        print(
            "        Sampler backend:  "
            + str(get_filepath("sampler_backend.h5"))
        )
        print(f"        Corner plot:          {get_filepath('corner.png')}")
        print(f"        SED plot:             {get_filepath('SED.png')}")
        print(f"        SED file:             {get_filepath('SED.txt')}")
        if args.mode == "phot":
            print(
                "        Model photometry:     "
                + str(get_filepath("model_photometry.txt"))
            )
        else:
            print(
                "        Model spectroscopy:   "
                + str(get_filepath("model_spectroscopy.txt"))
            )
        print(
            "        Output parameters:    "
            + str(get_filepath("parameters.txt"))
        )

    # Summarize convergence info if there were multiple objects.
    if len(convergence_times) > 1:
        convergence_times.sort(key=lambda x: x[1])
        if len(convergence_times) % 2 == 0:
            median_time = (
                convergence_times[len(convergence_times) // 2][1]
                + convergence_times[(len(convergence_times) + 1) // 2][1]
            ) / 2
        else:
            median_time = convergence_times[len(convergence_times) // 2][1]

        print_separator()
        print(f"  Median convergence time:  {median_time:.1f} minutes")
        print(
            "  Minimum convergence time: "
            f"ID {convergence_times[0][0]}, "
            f"{convergence_times[0][1]:.1f} minutes"
        )
        print(
            "  Maximum convergence time: "
            f"ID {convergence_times[-1][0]}, "
            f"{convergence_times[-1][1]:.1f} minutes"
        )

    print_separator(indent=4)
    print("Done, thanks for using NIFTY!")
