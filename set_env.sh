# Repo root + the LIBERO submodule (a PEP 420 namespace package that pip/uv
# can't expose, so it lives on PYTHONPATH rather than as an installed dep).
export PYTHONPATH=$PWD:$PWD/third_party/LIBERO

# numpy here is built on OpenBLAS, which defaults to one thread per core. The
# point-cloud deproject() is a stack of thin, memory-bound matmuls, so that
# default pins every core AND runs slower than single-threaded (~6.3ms vs ~4.0ms
# per 3-camera cloud on a 10-core mac). Cap it. ${VAR:-1} lets a caller override.
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}

source "$PWD/.venv/bin/activate"
