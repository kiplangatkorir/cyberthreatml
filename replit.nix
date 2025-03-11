{pkgs}: {
  deps = [
    pkgs.python3
    pkgs.pandoc
    pkgs.wkhtmltopdf
    pkgs.pango
    pkgs.harfbuzz
    pkgs.glib
    pkgs.fontconfig
    pkgs.tk
    pkgs.tcl
    pkgs.qhull
    pkgs.pkg-config
    pkgs.gtk3
    pkgs.gobject-introspection
    pkgs.ghostscript
    pkgs.freetype
    pkgs.ffmpeg-full
    pkgs.cairo
    pkgs.glibcLocales
  ];
}
