"""Parallel MPI map-reduce merge of DATM ha2x3h into consolidated forcing NetCDF."""

from parallel_pipeline.merge_datm.core import VAR_TO_DS_KEY, shard_files

__all__ = [
    "VAR_TO_DS_KEY",
    "main",
    "merge_datm_mpi",
    "shard_files",
]


def __getattr__(name: str):
    if name in ("main", "merge_datm_mpi"):
        from parallel_pipeline.merge_datm import runner

        return getattr(runner, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
