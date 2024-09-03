{
  pkgs ? import <nixpkgs> { },
}:
pkgs.mkShell {
  # nativeBuildInputs is usually what you want -- tools you need to run
  nativeBuildInputs = with pkgs.buildPackages; [
    python3
    python3Packages.jupyter
    python3Packages.openai
    python3Packages.black
    python3Packages.pyzmq
    python3Packages.prompt-toolkit
  ];

  shellHook = ''
    echo "~~ECHO SHELL~~"
    export OPENAI_API_KEY="$(cat ~/.config/echo/echo_openai_key)"
  '';

}
