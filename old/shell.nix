#
# $ nix-shell --pure --arg withBoost false --argstr tag 20.09
#

{
  tag ? "20.09",
  pymksVersion ? "cf653e004848c9c68ca31a85add0d1ac8611a93f"
}:
let
  pkgs = import (builtins.fetchTarball "https://github.com/NixOS/nixpkgs/archive/${tag}.tar.gz") {};
  pymkssrc = builtins.fetchTarball "https://github.com/materialsinnovation/pymks/archive/${pymksVersion}.tar.gz";
  pymks = pypkgs.callPackage "${pymkssrc}/default.nix" { graspi = null; };
  pypkgs = pkgs.python3Packages;
  hdfdict = pypkgs.buildPythonPackage rec {
    pname = "hdfdict";
    version = "0.3.1";
    src = pkgs.fetchurl {
      url="https://github.com/SiggiGue/hdfdict/archive/v${version}.tar.gz";
      sha256 = "sha256-+uAhoBktRYG8qFgkFsM6AjhgOYYME2XP8EFb5nKfWHs=";
    };
    propagatedBuildInputs = with pypkgs; [
      h5py
      pyyaml
    ];
  };
  extra = with pypkgs; [ black pylint flake8 ipywidgets zarr pymks h5py hdfdict ];

in
  (pymks.overridePythonAttrs (old: rec {

    propagatedBuildInputs = old.propagatedBuildInputs;

    nativeBuildInputs = propagatedBuildInputs ++ extra;

    postShellHook = ''
      export OMPI_MCA_plm_rsh_agent=${pkgs.openssh}/bin/ssh

      SOURCE_DATE_EPOCH=$(date +%s)
      export PYTHONUSERBASE=$PWD/.local
      export USER_SITE=`python -c "import site; print(site.USER_SITE)"`
      export PYTHONPATH=$PYTHONPATH:$USER_SITE
      export PATH=$PATH:$PYTHONUSERBASE/bin

      jupyter nbextension install --py widgetsnbextension --user > /dev/null 2>&1
      jupyter nbextension enable widgetsnbextension --user --py > /dev/null 2>&1
      pip install jupyter_contrib_nbextensions --user > /dev/null 2>&1
      jupyter contrib nbextension install --user > /dev/null 2>&1
      jupyter nbextension enable spellchecker/main > /dev/null 2>&1

      pip install --user nbqa
      pip install --user tqdm
      pip install --user modAL
    '';
  }))
