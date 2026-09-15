# ttnte

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Tests Status](https://github.com/myerspat/ttnte/actions/workflows/CI.yml/badge.svg)](https://github.com/myerspat/ttnte/actions/workflows)
[![VNV Status](https://github.com/myerspat/ttnte-vnv/actions/workflows/vnv.yml/badge.svg)](https://github.com/myerspat/ttnte-vnv/actions/workflows/vnv.yml)

`ttnte` is a Python library, backed by a compiled C++/CUDA extension, for
solving the multigroup discrete ordinates neutron transport equation (NTE) with
a discontinuous isogeometric analysis (IGA) spatial discretization. Operators
are assembled either as PyTorch sparse tensors or as tensor trains to exploit
the multiscale structure commonly found in reactor applications. The current
domain decomposition solver represents tensor trains with ttnte's own native C++
implementation (`ttnte.linalg.State` / `ttnte.linalg.Operator`, backed by
`ttnte.linalg.TTEngine`); the vendored
[torchTT](https://github.com/ion-g-ion/torchTT) library is still used internally
as one of a few interchangeable AMEn backends, and remains the tensor train
object (`torchtt.TT`) used by the deprecated global operator path. The IGA
discretization offers higher continuity than traditional finite elements and
benefits from working directly with CAD, cutting out the often expensive meshing
step. The broader goal is a general framework for studying tensor network
methods applied to the neutron transport equation.

## Solver approaches

`ttnte` is mid-migration between two solver designs:

- **Global operator (original design, deprecated):** every patch in the system
  is assembled into one monolithic TT linear system and solved together via
  `ttnte.driver.IGATransportDriver`, `ttnte.linalg.gmres()`, and
  `ttnte.linalg.power()`.
- **Domain decomposition (current design, in progress):** a fully parallel,
  non-overlapping upwind Schwarz-type domain decomposition. Each patch owns its
  own local linear system (`ttnte.linalg.LinearSystem`), and upwind boundary
  conditions are exchanged across patch interfaces. Local systems are solved
  with a `ttnte.solvers.LocalSolver` (e.g. `AMEnSolver`), scheduled as tasks on
  a DAG (`ttnte.task.TaskGraph`), and coordinated by a
  `ttnte.solvers.DDStrategy` (e.g. `BlockJacobiStrategy`). MPI communication and
  stream/thread pools are managed through `ttnte.parallel` (`ParallelContext`).

New solver, parallel, and task code should target the domain decomposition
model, not the global operator.

## Requirements

### System dependencies

- A C++20 compiler and [CMake](https://cmake.org/) (built via
  [scikit-build-core](https://scikit-build-core.readthedocs.io/))
- [Ninja](https://ninja-build.org/)
- An MPI implementation (e.g. OpenMPI: `openmpi-bin`, `libopenmpi-dev`)
- BLAS/LAPACK (e.g. `libblas-dev`, `liblapack-dev`)
- [CUDA toolkit](https://developer.nvidia.com/cuda-toolkit) (optional, only
  needed for GPU builds)

### Python dependencies

- [torch](https://pytorch.org/) `>=2.9.0`
- [torchtt](https://github.com/ion-g-ion/torchTT)
- [numpy](https://numpy.org/) `<2.0.0`
- [igakit](https://github.com/dalcinl/igakit)
- [geomdl](https://nurbs-python.readthedocs.io/en/5.x/)
- [pybind11](https://pybind11.readthedocs.io/en/stable/)
- [pandas](https://pandas.pydata.org/)
- [matplotlib](https://matplotlib.org/)
- [plotly](https://plotly.com/python/)
- [pyvista](https://docs.pyvista.org/)
- [h5py](https://docs.h5py.org/en/stable/index.html)
- [tqdm](https://github.com/tqdm/tqdm)
- [cotengra](https://cotengra.readthedocs.io/en/latest/index.html)
- [optuna](https://optuna.org/)
- [cmaes](https://pypi.org/project/cmaes/)
- [cotengrust](https://github.com/jcmgray/cotengrust)
- [cytools](https://cy.tools/)
- [kahypar](https://kahypar.org/)
- [loky](https://loky.readthedocs.io/en/stable/index.html)
- [networkx](https://networkx.org/)
- [opt_einsum](https://optimized-einsum.readthedocs.io/en/stable/)

`torch` and the other build requirements must already be installed in the active
environment before installing `ttnte`, since the C++ extension is built without
build isolation (`--no-build-isolation`) so it can link against the
Python-installed PyTorch.

## Installation

`ttnte` is a [scikit-build-core](https://scikit-build-core.readthedocs.io/)
project: the C++/CUDA extension is compiled as part of installation, so use the
provided `Makefile` rather than calling `pip` directly.

### Clone

```shell
git clone --recursive https://github.com/myerspat/ttnte.git && cd ttnte
```

The `--recursive` flag is required to pull in the git submodules under
`external/` (GKlib, METIS, ParMETIS, torchTT). If you already cloned without it,
run:

```shell
git submodule update --init --recursive
```

### Standard install

Installs a Release build with `-O3 -march=native` and CUDA enabled:

```shell
pip install "torch>=2.9.0"
pip install https://github.com/dalcinl/igakit/archive/refs/heads/master.zip
pip install git+https://github.com/ion-g-ion/torchTT.git
make install
```

### For developers

Installs an editable, Debug build with optimizations off and CUDA enabled, plus
the `dev` extras (`pytest`, `pre-commit`, etc.):

```shell
pip install "torch>=2.9.0"
pip install https://github.com/dalcinl/igakit/archive/refs/heads/master.zip
pip install git+https://github.com/ion-g-ion/torchTT.git
make dev
pre-commit install
```

### CPU-only install

If no CUDA toolkit is available (this is what CI uses), disable CUDA:

```shell
make dev USE_CUDA=OFF
```

### Makefile targets and overrides

```shell
make dev          # editable, Debug build, optimization OFF, CUDA ON  -> .[dev]
make install      # Release build, -O3 -march=native, CUDA ON
make clean        # remove build/, *.egg-info/, __pycache__
```

Common overrides, passed as `make <target> VAR=VALUE`:

- `JOBS=N`: parallel compile jobs (default `4`). Use `JOBS=$(nproc)` to use all
  available cores.
- `USE_CUDA=OFF`: build the CPU-only extension.
- `VERBOSE=ON`: verbose pip/CMake output.
- `BUILD_TYPE`, `TTNTE_OPTIMIZED`: override the build type and optimization
  defaults baked into each target.

The compiled module lands at `ttnte/cpp/ttnte_python.so`. C++ links against the
Python-installed PyTorch (its path is probed via
`torch.utils.cmake_prefix_path`), and the build must match its
`_GLIBCXX_USE_CXX11_ABI` setting; a mismatched torch build will cause link/ABI
errors at import time.

## Tests

```shell
pytest tests/unit/solvers                                  # one directory
pytest tests/unit/solvers/test_amen_solver.py               # one file
pytest tests/unit/solvers/test_amen_solver.py::test_name    # one test
mpirun -n 2 pytest --with-mpi tests/unit/parallel           # MPI tests
```

Tests exercise the compiled backend, so rebuild (`make dev`) after changing any
C++ before running them. `tests/regression/single_patch` and
`tests/regression/multi_patch` hold end-to-end solver regressions;
`tests/unit/<module>` mirrors the source module layout.

## Classes and Methods

- `ttnte.xs.Server`: class for handling multigroup cross section information.
- `ttnte.cad.Patch`: patch class.
- `ttnte.iga.IGAMesh`: meshing object for NURBS surfaces defined as
  `igakit.nurbs.NURBS`.
- `ttnte.assemblers.MatrixAssembler`: assembles the discretized system into
  `ttnte.assemblers.operators.SparseOperator`s.
- `ttnte.assemblers.TTAssembler`: assembles the discretized system into
  `torchtt.TT`s.
- `ttnte.linalg.LinearSystem`: per-patch local linear system (interior operator,
  boundary couplings, state, and source) used by the domain decomposition
  solvers.
- `ttnte.solvers.LocalSolver` / `ttnte.solvers.AMEnSolver`: solves one patch's
  local linear system, e.g. via AMEn.
- `ttnte.solvers.DDStrategy` / `ttnte.solvers.BlockJacobiStrategy`: builds the
  per-patch solve/communication task graph coordinating the domain decomposition
  solve.
- `ttnte.task.TaskGraph`: DAG of per-patch solve and communication tasks.
- `ttnte.parallel.ParallelContext` (`ttnte.parallel.mpi_context`): MPI context
  plus stream/thread pools used by the parallel solve.
- `ttnte.linalg.gmres()`: solves the resulting discretized fixed source system
  (global operator design).
- `ttnte.linalg.power()`: solves the resulting discretized eigenvalue problem
  (global operator design).
- `ttnte.linalg.LinearSolverOptions`: options class for GMRES used in
  `ttnte.linalg.power()`.

## Modules

- `ttnte.xs.benchmarks`: XS data sets from common neutron transport benchmarks.
- `ttnte.cad.curves`: methods for building NURBS curves used in the notebooks.
- `ttnte.cad.surfaces`: methods for building NURBS surfaces used in the
  notebooks.
- `ttnte.sources`: define fixed sources.

## Verification

You can find regression testing scripts used on published results at
[`ttnte-vnv`](https://github.com/myerspat/ttnte-vnv). This may also serve as
examples on how to use the code.
