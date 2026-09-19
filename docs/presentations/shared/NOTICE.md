# Third-party theme: neobeam

`neobeam.css` is vendored verbatim from
https://github.com/mikael-ros/neobeam (MIT License, Copyright (c) 2024 Embracket),
commit `ff0402c1bfcca181b7fc97f87a16e3d229a9277d` (2026-03-06).

It is vendored rather than hot-linked because the upstream README itself warns
that the direct link "might cause your presentation to change over time". To
update it, replace the file with `css/neobeam.css` from a newer commit and
record the new hash here; then rebuild every deck and look at it, because the
theme's colour derivation uses CSS relative-colour syntax that has changed
behaviour between Chromium versions.

Project-specific additions live in `surface-partition.css`, which imports this
file and is the theme every deck actually selects — `neobeam.css` itself is
never edited.

The theme imports Roboto, Roboto Mono and Noto Sans Math from Google Fonts at
build time; the exported PDF embeds them, so the deck is self-contained once
built, but building needs network access.
