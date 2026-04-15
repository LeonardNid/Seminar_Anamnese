{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    python312
    python312Packages.pip
    python312Packages.setuptools
    python312Packages.wheel

    # Systembibliotheken
    stdenv.cc.cc.lib
    zlib
    ffmpeg
  ];

  shellHook = ''
    # venv anlegen falls noch nicht vorhanden
    if [ ! -d ".venv" ]; then
      echo "Erstelle neues venv..."
      python3 -m venv .venv
    fi

    source .venv/bin/activate

    export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.zlib}/lib:${pkgs.ffmpeg.lib}/lib:$LD_LIBRARY_PATH"

    echo "Python-Umgebung aktiv: $(which python) ($(python --version))"
    echo "Pakete installieren:  pip install -r requirements.txt"
    echo "App starten:          streamlit run app.py"
  '';
}
