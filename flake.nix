{
  description = "python shell flake";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-23.05";
    flake-utils.url = "github:numtide/flake-utils";
    mach-nix.url = "github:davhau/mach-nix";
    pymks.url = "github:wd15/pymks/flakes";
  };

  outputs = { self, nixpkgs, mach-nix, flake-utils, pymks, ... }:
    let
      pythonVersion = "python310";
    in
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        snakemake = pkgs.snakemake;
        mach = mach-nix.lib.${system};
        pymks_ = pymks.packages.${system}.pymks;
        sfepy = pymks.packages.${system}.sfepy;

        pythonEnv = mach.mkPython {
          python = pythonVersion;
          packagesExtra = [ pymks_ sfepy ];

          providers.jupyterlab = "nixpkgs";

          requirements = ''
            tqdm
            jupytext
            papermill
            ipywidgets
            scikit-learn
            dask
            ipdb
            setuptools
            pot
          '';
        };
      in
      {
        devShells.default = pkgs.mkShellNoCC {
          packages = [ pythonEnv snakemake ];

          shellHook = ''
            export PYTHONPATH="${pythonEnv}/bin/python"

            SOURCE_DATE_EPOCH=$(date +%s)
            export PYTHONUSERBASE=$PWD/.local
            export USER_SITE=`python -c "import site; print(site.USER_SITE)"`
            export PYTHONPATH=$PYTHONPATH:$USER_SITE
            export PATH=$PATH:$PYTHONUSERBASE/bin

            jupyter serverextension enable jupytext
            jupyter nbextension install --py jupytext --user
            jupyter nbextension enable --py jupytext --user
          '';
        };
      }
    );
}
