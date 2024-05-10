{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };
  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem
      (system:
        let
          pkgs = import nixpkgs {
            inherit system;
          };
        in
        with pkgs;
        {
          devShells.default = (mkShell.override { stdenv = gcc12Stdenv; }) {
            buildInputs = [
              brotli
              (abseil-cpp.override { stdenv = gcc12Stdenv; })
              gcc12
              pkg-config
              (pkgs.python311.withPackages
                (ps: with ps; [ pybind11 ]))
            ];
          };
        }
      );
}
