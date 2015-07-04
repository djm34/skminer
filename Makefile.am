# allow to use Host cuda functions in C/C++
DEF_INCLUDES = @CUDA_INCLUDES@



EXTRA_DIST		= autogen.sh README.txt LICENSE.txt \
			  cudaminer.sln cudaminer.vcxproj cudaminer.vcxproj.filters \
			  cpuminer-config.h.in

bin_PROGRAMS = skminer

skminer_SOURCES	= elist.h miner2.h compat.h bignum.h core.h cuda_helper.h \
		type.h uint256.h util_llh.cpp util_llh.h \
		cpu-miner.cpp heavy/heavy.cu  \
 	   	Algo/sk1024_keccak.cu Algo/cuda_skein1024.cu Algo/sk1024.cu \
		hash/brg_endian.h hash/brg_types.h hash/CBlock.cpp hash/CBlock.h hash/Connection.cpp \
		hash/Connection.h hash/crypto_hash.h hash/DDOS_Filter.cpp hash/DDOS_Filter.h \
		hash/DDOS_Score.cpp hash/DDOS_Score.h hash/Keccak-compact64.c hash/KeccakDuplex.c \
		hash/KeccakDuplex.h hash/KeccakF-1600-interface.h hash/KeccakHash.c hash/KeccakHash.h \
		hash/KeccakSponge.c hash/KeccakSponge.h hash/Miner.cpp hash/Miner.h hash/MinerThread.cpp \
		hash/MinerThread.h hash/Outbound.cpp hash/Outbound.h hash/Packet.cpp hash/Packet.h \
		hash/skein_block.cpp hash/skein.cpp hash/skein.h hash/skein_iv.h hash/skein_port.h \
		hash/templates.h hash/Timer.cpp hash/Timer.h hash/uint1024.h


if HAVE_NVML
nvml_defs = -DUSE_WRAPNVML
nvml_libs = -ldl
endif



skminer_LDFLAGS  = $(PTHREAD_FLAGS) @CUDA_LDFLAGS@
skminer_LDADD    = @PTHREAD_LIBS@ @WS2_LIBS@ @CUDA_LIBS@ @OPENMP_CFLAGS@ @LIBS@ @BOOST_LIB@ $(nvml_libs)
skminer_CPPFLAGS = @OPENMP_CFLAGS@ $(CPPFLAGS) $(PTHREAD_FLAGS) $(BOOST_FLAGS) -fno-strict-aliasing $(BOOST_INCLUDES)  $(DEF_INCLUDES) $(nvml_defs) 

nvcc_ARCH  = -gencode=arch=compute_50,code=\"sm_50,compute_50\"
#nvcc_ARCH += -gencode=arch=compute_52,code=\"sm_52,compute_52\"
#nvcc_ARCH += -gencode=arch=compute_35,code=\"sm_35,compute_35\"
#nvcc_ARCH = -gencode=arch=compute_30,code=\"sm_30,compute_30\"
#nvcc_ARCH = -gencode=arch=compute_20,code=\"sm_21,compute_20\"

nvcc_FLAGS = $(nvcc_ARCH) @CUDA_INCLUDES@ -I. @CUDA_CFLAGS@
nvcc_FLAGS += --ptxas-options="-v"

# we're now targeting all major compute architectures within one binary.
.cu.o:
	$(NVCC) $(nvcc_FLAGS)  -o $@ -c $<


