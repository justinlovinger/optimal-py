{ pkgs ? (import (builtins.fetchGit {
  url = https://github.com/NixOS/nixpkgs.git;
  ref = "nixos-20.09";
  rev = "df8e3bd110921621d175fad88c9e67909b7cb3d3";
}) {}) }:

let
  pythonPackages = pkgs.python2Packages;
in pkgs.pythonPackages.buildPythonPackage {
  pname = "optimal";
  version = "0.2.0";
  src = pkgs.lib.cleanSource ./.;

  checkInputs = with pythonPackages; [ pytest ];
  propagatedBuildInputs = with pythonPackages; [
    dill
    numpy
  ];

  doCheck = false; # Many tests are stochastic and may fail.
  checkPhase = ''pytest -m "not slowtest"'';

  meta = with pkgs.stdenv.lib; {
    description = "A python metaheuristic optimization library";
    homepage = "https://github.com/JustinLovinger/optimal";
    license = licenses.mit;
    maintainers = [ ];
  };
}
