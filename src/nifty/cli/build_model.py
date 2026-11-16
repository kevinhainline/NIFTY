"""CLI for building model grid files."""

import argparse
from pathlib import Path

import numpy as np

from ..build_model import (
    build_atmo2020,
    build_lowz,
    build_sonora_diamondback_cloudless,
    build_sonora_diamondback_cloudy,
    build_sonora_elf_owl,
    build_sonora_flame_skimmer_disequilibrium,
    build_sonora_flame_skimmer_equilibrium,
    build_sonora_ph3,
    load_filters,
)
from ..model import fill_model_fit
from .print import print_banner, print_filters

BUILD_FUNCTION = {
    "SonoraFlameSkimmerDeq": build_sonora_flame_skimmer_disequilibrium,
    "SonoraFlameSkimmerEq": build_sonora_flame_skimmer_equilibrium,
    "SonoraDiamondbackCloudy": build_sonora_diamondback_cloudy,
    "SonoraDiamondbackCloudless": build_sonora_diamondback_cloudless,
    "SonoraElfOwl": build_sonora_elf_owl,
    "SonoraElfOwlPH3": build_sonora_ph3,
    "ATMO2020": build_atmo2020,
    "LOWZ": build_lowz,
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NIFTY ModelGrid builder.")
    parser.add_argument(
        "model",
        choices=list(BUILD_FUNCTION.keys()),
        help="Name of the model to build a grid",
    )
    parser.add_argument(
        "--path",
        help="Path to the model file or directory (see docstring for layout details)",
        required=True,
        dest="model_path",
    )
    parser.add_argument(
        "--config",
        help="Path to the NIFTY filter config JSON file",
        required=True,
        dest="config_path",
    )
    parser.add_argument(
        "--output_raw",
        help=(
            "Output filename for raw unfilled model. Will not be saved if "
            "model is already complete. Defaults to "
            "{model}_raw_ModelGrid.tar.gz"
        ),
        dest="output_raw_path",
    )
    parser.add_argument(
        "--output_completed",
        help=(
            "Output filename for complete model. Defaults to "
            "{model}_complete_ModelGrid.tar.gz"
        ),
        dest="output_complete_path",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    model = args.model
    if args.output_raw_path is None:
        output_raw_path = Path(f"{model}_raw_ModelGrid.tar.gz")
    else:
        output_raw_path = Path(args.output_raw_path)
    if args.output_complete_path is None:
        output_complete_path = Path(f"{model}_complete_ModelGrid.tar.gz")
    else:
        output_complete_path = Path(args.output_complete_path)

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Error: Model path {model_path} does not exist.")
        return 1

    config_path = Path(args.config_path)
    if not config_path.exists():
        print(f"Error: Config path {config_path} does not exist.")
        return 1

    print_banner("Model Builder")
    print(f"Model            : {model}")
    print(f"Path             : {model_path}")
    print(f"Config           : {config_path}")
    print(f"Output Raw       : {output_raw_path}")
    print(f"Output Complete  : {output_complete_path}")
    print(" - - - - - - - - ")

    print(f"Opening config/filters JSON: {config_path}")
    filters, filter_names = load_filters(config_path)
    print_filters(filter_names)

    model_grid = BUILD_FUNCTION[model](
        model_path,
        filters=filters,
        filter_names=filter_names,
    )

    if np.any(np.isnan(model_grid.phot)) or np.any(np.isnan(model_grid.spec)):
        print("Saving raw model grid.")
        model_grid.save(output_raw_path)
        print("Filling model grid.")
        filled_grid = fill_model_fit(model_grid, progress_bar=True)
        print("Saving complete model grid.")
        filled_grid.save(output_complete_path)
    else:
        print("Model grid is complete.")
        print("Saving complete model grid.")
        model_grid.save(output_complete_path)

    print("Done!")
