{ pkgs ? import <nixpkgs> {} }:

let
  lib-path = pkgs.lib.makeLibraryPath [
    pkgs.stdenv.cc.cc.lib
    pkgs.zlib
    pkgs.ffmpeg.lib
  ];
in
pkgs.mkShell {
  buildInputs = with pkgs; [
    python3
    python3Packages.pip
    python3Packages.virtualenv
    stdenv.cc.cc.lib
    zlib
    ffmpeg
  ];

  shellHook = ''
    export LD_LIBRARY_PATH="${lib-path}:$LD_LIBRARY_PATH"
    
    echo "Python environment is ready."
    echo "To get started:"
    echo "1. python -m venv .venv"
    echo "2. source .venv/bin/activate"
    echo "3. pip install -r requirements.txt"
    echo "4. streamlit run app.py"
  '';
}
