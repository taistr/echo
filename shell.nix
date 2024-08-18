{ pkgs ? import <nixpkgs> {} }:
  pkgs.mkShell {
    # nativeBuildInputs is usually what you want -- tools you need to run
    nativeBuildInputs = with pkgs.buildPackages; [ 
      python3
      jupyter
      openai
     ];

  shellHook =
  ''
    echo "~~ECHO SHELL~~"
    export OPENAI_API_KEY="$(cat ~/.config/echo/echo_openai_key)"
  '';

     
}