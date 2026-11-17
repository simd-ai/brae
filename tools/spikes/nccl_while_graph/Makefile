# make NCCL_ROOT=/path/to/nccl [CUDA_HOME=/usr/local/cuda] [ARCH=70] [MPI_HOME=/usr/lib64/openmpi]
# HOSTCXX=g++-14 selects the nvcc host compiler (nvcc 12.x supports gcc <= 14).
# NCCL_ROOT holds include/nccl.h and lib/libnccl.so* (a system install, the tarball, or the nvidia-nccl-cu12 wheel dir).
CUDA_HOME ?= $(dir $(shell which nvcc))..
ARCH      ?= 70
NCCL_ROOT ?= $(CUDA_HOME)
NVCC      ?= $(CUDA_HOME)/bin/nvcc
HOSTCXX   ?=
FLAGS_HOST = $(if $(HOSTCXX),-ccbin $(HOSTCXX))
FLAGS     = $(FLAGS_HOST) -O2 -std=c++17 -gencode arch=compute_$(ARCH),code=sm_$(ARCH) --fmad=false -Xcompiler -ffp-contract=off \
            $(if $(shell grep -l graphUsageMode $(NCCL_ROOT)/include/nccl.h 2>/dev/null),-DHAVE_GRAPH_USAGE_MODE) -I$(NCCL_ROOT)/include -L$(NCCL_ROOT)/lib -Xlinker -rpath,$(NCCL_ROOT)/lib $(EXTRA)
NCCLLIB   = $(or $(wildcard $(NCCL_ROOT)/lib/libnccl.so),-l:libnccl.so.2)

all: nccl_spike $(if $(MPI_HOME),nccl_spike_mpi)

nccl_spike: nccl_spike.cu
	$(NVCC) $(FLAGS) $< -o $@ $(NCCLLIB)

nccl_spike_mpi: nccl_spike.cu
	$(NVCC) $(FLAGS) -DSPIKE_MPI -I$(MPI_HOME)/include -L$(MPI_HOME)/lib -Xlinker -rpath,$(MPI_HOME)/lib $< -o $@ $(NCCLLIB) -lmpi

clean:
	rm -f nccl_spike nccl_spike_mpi
