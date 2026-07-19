<img src="nifty_logo_zoom.png" alt="NIFTY logo" width="300"/>

# NIFTY
*Near-Infrared Fitting for T and Y Dwarfs*

*Kevin Hainline, Jake Helton, and Evan Chen*

NIFTY is a code designed to fit JWST NIRCam/MIRI photometry or NIRSpec prism 
spectroscopy of cold brown dwarf candidates with the LOWZ (Meisner et al. 2021), 
ATMO2020 (Phillips et al. 2020), Sonora Elf Owl v2 (Mukherjee et al. 2024, 
Wogan et al. 2025), or earlier Sonora Elf Owl (with PH3, Beiler et al. 2024) atmospheric models, using 
a Bayesian framework with the emcee sampler (https://emcee.readthedocs.io/en/stable/user/sampler/) 
or the Nautilus sampler (https://nautilus-sampler.readthedocs.io/en/latest/). 
This code was described in Hainline et al. (2026) (doi.org/10.48550/arXiv.2510.00111).

NIFTY operates in two modes:

- `phot` — fit broadband photometry from a catalog of one or more sources.
- `spec` — fit a single NIRSpec prism spectrum.

In both modes, NIFTY produces a corner plot, a best-fit SED or spectrum plot, 
and a text file with the 16th, 50th, and 84th percentiles of all fit parameters.
Photometry fits typically converge in under 3 minutes per source for ~14 
bands, though convergence time varies, especially at high SNR. Spectroscopic 
fits are slightly longer.

## Installation

The installation comes with the NIFTY Python package and the commands `nifty` 
and `build_model`.

The easiest way to install NIFTY is through pip:
```
pip install astro-nifty
```
The main pip installation only includes the dependencies required for sampling. 
If you also want to build your own models, install the optional dependencies:
```
pip install astro-nifty[build-model]
```
If you are installing with pip, it is recommended that you install within a 
**virtual environment** to keep dependencies clean. See the
[Python tutorial](https://docs.python.org/3/tutorial/venv.html) on virtual 
environments and packages for more information.

Alternatively, you can clone the NIFTY repository and set up an environment 
using conda or micromamba:
```
conda env create -f environment.yml
conda activate NIFTY
```

## Running NIFTY

The easiest way to run NIFTY is through the `nifty` command.
NIFTY runs in two modes: photometry mode and spectroscopy mode, which require 
slightly different arguments.

To see the available arguments, you can run: `nifty phot -h` or `nifty spec -h` 
to display the help message for that mode.

### Photometry mode (`nifty phot`)

Fit broadband photometry for one or more sources from a catalog.

**Single source:**
```
nifty phot \
  --model path/to/model_grid.tar.gz \
  --stub JADES-GS \
  --config BD_NIRCam_MIRI_filters.json \
  --id 20541 \
  path/to/photometry_file.fits
```

**List of IDs from a file** (one ID per line):
```
nifty phot \
  --model path/to/model_grid.tar.gz \
  --stub JADES-GS \
  --config BD_NIRCam_MIRI_filters.json \
  --idlist all_source_IDs.dat \
  path/to/photometry_file.fits
```

**Multiple IDs:**
```
nifty phot \
  --model path/to/model_grid.tar.gz \
  --stub JADES-GS \
  --config BD_NIRCam_MIRI_filters.json \
  --id 20541 --id 452029 --id 430165 \
  path/to/photometry_file.fits
```

### Spectroscopy mode (`nifty spec`)

Fit a single NIRSpec prism spectrum. The spectrum file should be a
whitespace-delimited text file with three columns: wavelength (microns), 
flux (nJy), and flux error (nJy). The `--id` argument is used only to
label output files.

```
nifty spec \
  --model path/to/model_grid.tar.gz \
  --stub JADES-GS \
  --id 20541 \
  path/to/spectrum.txt
```

### Optional arguments

| Argument | Description |
|---|---|
| `--output` | Output folder (default: `<Model>_output/`) |
| `--frac_model_floor` | Fractional model flux floor added in quadrature to photometry errors, to account for model systematics (e.g. `0.03` for 3%). Photometry mode only; no floor is applied in spectroscopy mode. |
| `--sampler` | Sampler used to estimate the posterior. Currently supports `mcmc` and `nautilus` |

## Configuration File (`--config`)

NIFTY uses a single JSON file to describe all filter information needed for 
photometry fitting. Here is a sample entry:

```json
{
  "filter_columns": {
    "JWST_F115W": {
      "extension": "CIRC",
      "flux": "F115W_CIRC3",
      "error": "F115W_CIRC3_e",
    },
    "JWST_F444W": {
      "extension": "CIRC",
      "flux": "F444W_CIRC3",
      "error": "F444W_CIRC3_e",
    }
  }
}
```
The filter name must match a [valid sedpy filter](https://github.com/bd-j/sedpy/blob/main/sedpy/data/filters/README.md).
If no telescope prefix is provided, the prefix is assumed to be "JWST_".

- `extension`: the FITS HDU name where this filter's data lives. Ignored for 
plain-text catalogs.
- `flux` / `error`: exact column names in the photometry file.

An example config file for JADES NIRCam + MIRI observations is included as
`BD_NIRCam_MIRI_filters.json`.

## Photometry File Format

NIFTY can read:

- **FITS** files, with filter fluxes and errors in named HDU extensions and 
columns as specified in the config JSON.
- **Plain-text** catalogs, whitespace-delimited, one row per object.

For plain-text catalogs, include an `ID` column (case-sensitive) and flux/error
columns whose names match the config JSON exactly. Do not prefix the header 
line with `#`. Example:

```
ID     F115W_CIRC3     F115W_CIRC3_e     F444W_CIRC3     F444W_CIRC3_e
101    0.123           0.010             0.456           0.025
```

All fluxes and errors should be in nJy. A minimum relative flux error of 5% is 
applied automatically; errors below this floor are inflated to `0.05 * flux`.

## Output Files

For each fitted source, NIFTY writes the following files to the output directory:

Each file name is prefixed with `<stub>_<ID>_<Model>_`.
| File | Description |
|---|---|
| `sampler_backend.h5` | Backend file for sampler (e.g. MCMC chains) |
| `corner.png` | Corner plot of the posterior |
| `SED.png` | Best-fit SED or spectrum plot |
| `SED.txt` | Model spectrum envelope (16th, 50th, 84th percentile flux, in nJy, vs wavelength in microns) |
| `parameters.txt` | 16th, 50th, and 84th percentile parameter values, plus chi-square. Distance is in parsecs. |
| `model_photometry.txt` / `model_spectroscopy.txt` | 50th percentile model fit for either spectroscopy or photometry depending on mode.|

## Model Parameters

| Model | Parameters fit |
|---|---|
| `SonoraElfOwl` | Teff, log(g), log(Kzz), [M/H], C/O, distance (pc) |
| `SonoraElfOwlPH3` | Teff, log(g), log(Kzz), [M/H], C/O, distance (pc) |
| `ATMO2020` | Teff, log(g), [M/H], distance (pc) |
| `LOWZ` | Teff, log(g), log(Kzz), [M/H], C/O, distance (pc) |

All fits assume a source radius of 1 Jupiter radius. Distances are always
reported in parsecs.


## Creating the ModelGrid Files

Before running NIFTY, download the model grids you want to fit against, then 
run `build_model` to build the model grid file that NIFTY uses during fitting. 

For a more detailed description of parameters, run `build_model -h`.

```
build_model \
    --path path/to/model/files/ \
    --config BD_NIRCam_MIRI_filters.json \
    ModelName
```

The valid model names are `SonoraElfOwl`, `SonoraElfOwlPH3`, `ATMO2020`, `LOWZ`.

Typically, there are two ModelGrid files outputted. The **raw grid** is the 
grid created by reading in all the points that are part of the model. However, 
many models do not have values at all grid points, so missing points are filled. 
with NaN. The **complete grid** has  values for all grid points. If the raw 
grid is missing values, they are filled by a fitting procedure (see code for 
details). If the raw grid is already complete, only the complete grid is saved.

By default, the output model grid files are named 
`<Model>_raw_ModelGrid.tar.gz` and `<Model>_complete_ModelGrid.tar.gz`.

The Sonora Elf Owl model grid spans several tens of GB and can take a few hours
to process. The other grids are considerably smaller.

### Expected Model File Layouts

**SonoraElfOwl** — directory of temperature-range tarballs containing NetCDF 
files:
```
Sonora_Elf_Owl_v2/
    teff_275_325.tar.gz
    teff_350_400.tar.gz
    ...
```

**SonoraElfOwlPH3** — a single `.npz` file:
```
elf_owl_disequilibrium_PH3.npz
```

**ATMO2020** — per-metallicity subdirectories containing `.dat` files:
```
meisner_2023/
    grid_m1.0/
        spec_jwst_t700_g3.5_m1.0_kg_g3.5.dat
        ...
    grid_m0.5/
    grid_p0/
    grid_p0.3/
```

**LOWZ** — CSV index and models tarball
```
LOWZ/
    LOWZ_models_index.csv
    models.tar.gz
```

## References

- Meisner et al. 2021 (LOWZ): https://doi.org/10.3847/1538-4357/ac013c
- Phillips et al. 2020 (ATMO2020): https://doi.org/10.1051/0004-6361/201937381 
- Mukherjee et al. 2024 (Sonora Elf Owl): https://doi.org/10.3847/1538-4357/ad18c2
- Wogan et al. 2025 (Sonora Elf Owl v2): https://doi.org/10.3847/2515-5172/add407
- Beiler et al. 2024: https://doi.org/10.5281/zenodo.11370829
- Foreman-Mackey et al. 2013 (emcee): https://doi.org/10.1086/670067
- Lange 2023 (nautilus): https://doi.org/10.1093/mnras/stad2441
