"""Pretty printing functions for the NIFTY CLI."""

from collections.abc import Sequence

from ..model import ModelGrid

__all__ = [
    "print_banner",
    "print_filters",
    "print_id",
    "print_model_grid_range",
    "print_separator",
]

# Axis key to readable name.
AXIS_LABEL = {
    "teff": "Teff",
    "logg": "log(g)",
    "kzz": "kzz",
    "mh": "[M/H]",
    "co": "C/O",
}

# Model key to readable name.
MODEL_LABEL = {
    "SonoraElfOwl": "Sonora Elf Owl",
    "SonoraElfOwlPH3": "Sonora Elf Owl + PH3",
    "ATMO2020": "ATMO2020",
    "LOWZ": "LOWZ",
}


def print_banner(sub: str | None = None) -> None:
    """Print the NIFTY banner.

    Parameters
    ----------
    sub : str, optional
        The subtitle for this specific NIFTY tool (e.g. Model Builder).
    """
    if sub is None:
        sub = ""
    print()
    print("▗▖  ▗▖▗▄▄▄▖▗▄▄▄▖▗▄▄▄▖▗▖  ▗▖   Near-Infrared")
    print("▐▛▚▖▐▌  █  ▐▌     █   ▝▚▞▘    Fitting for")
    print("▐▌ ▝▜▌  █  ▐▛▀▀▘  █    ▐▌     T and Y Dwarfs")
    print(f"▐▌  ▐▌▗▄█▄▖▐▌     █    ▐▌     {sub}")
    print()
    print("https://github.com/kevinhainline/NIFTY")


def print_filters(filter_names: Sequence[str]) -> None:
    """Print a list of filters names.

    A maximum of six filter names are printed per line.

    Parameters
    ----------
    filter_names : sequence of str
        Names of the filters to print.
    """
    # Display filter names in uppercase and remove any prefix (e.g. "JWST_").
    filter_names = [name.upper().split("_")[-1] for name in filter_names]
    print("    Using " + str(len(filter_names)) + " filters:")
    line = "       " + filter_names[0]
    for i in range(1, len(filter_names)):
        if not i % 6:
            line += "\n       " + filter_names[i]
        else:
            line += ", " + filter_names[i]
    print(line)
    print(" - - - - - - - - ")


def print_model_grid_range(model_grid: ModelGrid, *, indent: int = 2) -> None:
    """Print the parameter range for a model grid.

    Parameters
    ----------
    model_grid : ModelGrid
        Model grid to print parameter ranges.
    indent : int, default=2
        Number of spaces to indent the message.
    """
    indent_str = " " * indent
    for name, axis in zip(model_grid.axes, model_grid.points):
        if name == "teff":
            # Convert log T_eff to linear T_eff without modifying the original.
            axis = 10**axis
        print(
            f"{indent_str}{AXIS_LABEL.get(name, name)}: "
            f"{axis[0]:.2f} to {axis[-1]:.2f}"
        )


def print_separator(n: int = 8, indent: int = 0) -> None:
    """Print a horizontal separator.

    Parameters
    ----------
    n : int, default=8
        Number of times to repeat the separator pattern.
    indent : int, default=0
        Number of extra spaces to indent on the left.
    """
    print(" " * indent + " -" * n)


def print_id(obj_id: int, model_name: str) -> None:
    """Print the object ID header before fitting an object.

    Parameters
    ----------
    obj_id : int
        Object ID to print.
    model_name : str
        Standard name of the model.
    """
    PATTERN = " *"
    REPEAT = 8
    obj_id_label = f"OBJECT ID {obj_id}"
    model_label = f"{MODEL_LABEL.get(model_name, model_name)}"
    labels = [obj_id_label, model_label]
    # The label with the maximum value determines the boundaries.
    max_len = max(len(label) for label in labels)
    # We want the length to always be odd so that an extra space is added for
    # the best alignment.
    if max_len % 2 == 0:
        max_len += 1

    border = PATTERN * (2 * REPEAT + (max_len + 1) // 2)
    print(border)
    for label in labels:
        diff = max_len - len(label)
        # The left will get an extra star over the right if needed.
        # 2 * left_extra + 2 * right_extra + extra_space == diff.
        left_extra = (diff + 2) // 4
        right_extra = diff // 4
        extra_space = diff % 2
        print(
            f"{PATTERN * (REPEAT + left_extra)}{' ' * extra_space} "
            f"{label}{PATTERN * (REPEAT + right_extra)}"
        )
    print(border)
