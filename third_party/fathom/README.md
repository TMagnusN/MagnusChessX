# Fathom

This directory contains the files required to build the Fathom Syzygy
tablebase probe:

- Upstream: https://github.com/jdart1/Fathom
- Commit: `c9c6fef0dddc05d2e242c183acf5833149ab676d`
- License: MIT, reproduced in `LICENSE`

The code is compiled as C and wrapped by `src/Syzygy.cpp`. Stockfish was used
only as a behavioral reference for option semantics and search integration;
no GPL Stockfish source is included.
