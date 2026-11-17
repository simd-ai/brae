# Frozen-state multi-rank I/O harness

`tests/mgpu_frozen.cu` (target `mgpu_frozen`) drives everything of interFoam that is not the numerical step on N virtual
ranks (ProcComm loopback, one thread per rank, one GPU; np=1 runs with no communicator = the single-GPU path) from a
state OpenFOAM made, and `tools/mgpu_oracle` holds the references.

```
tools/mgpu_oracle/frozen_state.sh tank 4 scotch 5     # -> /tmp/brae-frozen/tank_np4/
   serial/      OpenFOAM serial run, 5 steps (dumpInterFoam)       dec/  decomposePar of its last time + references
   dec/postProcessing  postProcess -parallel on the state (17 digits)   dec/inrun  serial's own FOs (12 digits)
   dec_par/     the state of OpenFOAM's own -parallel run (coupled processor values)   dec_par/inrun its FOs
   recon/       reconstructPar of dec
mgpu_frozen globals <dec> <time>         deltaN, needReference, pRefPoint owner, end statistics: same on every rank
mgpu_frozen fo      <dec> <time> <out>   function objects (BRAE_FO_CONTROLDICT=dec/system/controlDict.run for the in-run refs)
   compare_fo.py dec/postProcessing <out>/postProcessing [--last-row for dec/inrun]
mgpu_frozen write   <dec_par> <time> <out> [ascii|binary] [nWrites] [purgeWrite]    per-rank writer
   compare_rank_dirs.py dec_par <out> <time> --ignore=uniform/...     (byte compare after the banner)
mgpu_frozen procvals <dec_par> <time>    processor face values from ProcComm halo + processorFvPatch weights vs stored
```
Set `BRAE_FO_CONTROLDICT` as above for `write` too (writePrecision of the runs). The GPU is shared: a CUDA out of
memory in `cudaEventCreate` is not a failure, retry.

Measured (tank, 5 steps): see /tmp/WriterFO/REPORT.md.
