<div align="center">

  ![MagnusChessX logo](logo.png)

  [![C++20][cpp-badge]][repo-link]
  [![UCI][uci-badge]][repo-link]
  [![MNUE][mnue-badge]][mnue-link]
  [![Syzygy][syzygy-badge]][repo-link]
  [![License][license-badge]][license-link]
  [![Commits][commits-badge]][commits-link]

</div>

MagnusChessX is a C++20 UCI chess engine by Theodore Magnus &Oslash;en Nidhar.

Its current playable line, **Thinking**, is a deliberately modest reference engine: low-strength, rudimentary in positional understanding, and limited in practical competitiveness. The point is not to pretend it is already a tournament engine; the point is to keep a clean engine body for testing search behavior, evaluation design, and MNUE ideas.

MagnusChessX communicates over the Universal Chess Interface, which means you can talk to it in the terminal,
or install it into a UCI-compatible graphical user interface, like [En Croissant](https://encroissant.org/download)
or [Nibbler](https://github.com/rooklift/nibbler).

The default network line is maintained separately in [MagnusMNUE](https://github.com/TMagnusN/MagnusMNUE). The normal evaluation path is P2 MNUE; an experimental X2-K16-pawn MNUE path is also present because its structure is unusual enough to be worth testing, even when it is not the default strength target.

MagnusChessX does not support Chess960. It probes Syzygy tablebases up to 7 pieces through the vendored [Fathom](https://github.com/jdart1/Fathom) backend.

PS: part of the MNUE design was learned from Reckless.

## Evaluation Development History/Originality (HCE/MNUE)

MagnusChessX began with a hand-written evaluation function. That was replaced by a simple Chess768 NNUE, and later by the current P2 MNUE line. The MNUE architecture is designed for MagnusChessX and trained with [bullet](https://github.com/jw1912/bullet).

Current training data is generated with Stockfish tooling. The project does not claim original self-play-only training data.

Testing is kept consistent: regular engine tests use `UHO_4060_v4.epd` and a private OpenBench instance.

## Thanks and Acknowledgements

[Fathom](https://github.com/jdart1/Fathom), used for Syzygy tablebase probing.

[bullet](https://github.com/jw1912/bullet), used for neural-network training.

Reckless, for ideas that influenced parts of the MNUE design.

MagnusChessX is distributed under the MIT License. See `LICENSE`.

[repo-link]: https://github.com/TMagnusN/MagnusChessX
[mnue-link]: https://github.com/TMagnusN/MagnusMNUE
[commits-link]: https://github.com/TMagnusN/MagnusChessX/commits/main
[license-link]: https://github.com/TMagnusN/MagnusChessX/blob/main/LICENSE

[cpp-badge]: https://img.shields.io/badge/C%2B%2B-20-blue?style=for-the-badge
[uci-badge]: https://img.shields.io/badge/UCI-supported-brightgreen?style=for-the-badge
[mnue-badge]: https://img.shields.io/badge/MNUE-P2%20default-purple?style=for-the-badge
[syzygy-badge]: https://img.shields.io/badge/Syzygy-7--piece-informational?style=for-the-badge
[license-badge]: https://img.shields.io/github/license/TMagnusN/MagnusChessX?style=for-the-badge&label=license&color=success
[commits-badge]: https://img.shields.io/github/commits-since/TMagnusN/MagnusChessX/latest?style=for-the-badge
