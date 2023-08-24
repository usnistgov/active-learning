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
        modal = mach.buildPythonPackage {
          python = pythonVersion;
          name = "modal";
          version = "0.4.2.1";
          src = pkgs.pythonPackages.fetchPypi {
            version = "0.4.2.1";
            pname = "modAL-python";
            hash = "sha256-sYTa0Ve7xY+S5oiDWkiF/UhBMxJmHBokuVdbz+6dmtE=";
          };
        };


        pythonEnv = mach.mkPython {
          python = pythonVersion;
          packagesExtra = [ pymks_ sfepy modal ];


          providers.jupyterlab = "nixpkgs";

          requirements = ''
            tqdm
            jupytext
            papermill
            hdfdict
            ipywidgets
            dask-ml==1.6.0
            scikit-learn==1.0.2
            dask==2.22.0
            ipdb
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
