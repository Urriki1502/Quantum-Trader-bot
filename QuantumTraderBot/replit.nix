{pkgs}: {
  deps = [
    pkgs.glibcLocales
    pkgs.libyaml
    pkgs.rustc
    pkgs.pkg-config
    pkgs.libxcrypt
    pkgs.libiconv
    pkgs.cargo
    pkgs.postgresql
    pkgs.openssl
  ];
}
