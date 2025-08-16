all: hgemm

hgemm : hgemm.cu cublaslt_hgemm.cuh
	nvcc -o hgemm hgemm.cu -O2 -arch=sm_86 -std=c++17 -Iexternal/include --expt-relaxed-constexpr -cudart shared --cudadevrt none -lcublasLt

clean:
	rm -rf hgemm