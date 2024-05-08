# Active Learning

Repository for work related to Liu et al. "Active Learning Using Various Representations." Digital Discovery (in peer review, submitted 03/09/2024)​

# Installation

## Conda

Install
[Mamba](https://mamba.readthedocs.io/en/latest/micromamba-installation.html#umamba-install). The
environment is contained in the `environment.yml` file. It's name is
`active-learning` so remove it if it already exists.

    $ mamba env remove -n active-learning

Then create the new environment

    $ mamba env create -f environment.yml
    $ mamba activate active-learning

To test that things are working.

    $ cd 2D

Edit `config.yml` and set `n_iterations: 1` so that the job runs
quickly and then run

    $ snakemake --cores 1

It should generate a file called `plot.png` in the `job_name`
directory defined in `config.yml`. Check that the plot looks
reasonable.
