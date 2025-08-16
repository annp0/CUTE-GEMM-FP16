# CuTe-Gemm-Fp16

HGEMM for Ampere Architecture and above (sm >= 80). The basic idea is the same as my [previous kernel](https://github.com/annp0/GEMM-FP16), but instead of using the WMMA API, I used CuTe templates. This allows us to XOR swizzle shared memory and reduce bank conflicts. I have also written an epilogue for this kernel where it first permutates the output in shared memory so that we can use vectorized transfer back to global memory.

The benchmark was done on an NVIDIA A6000 (`void hgemm<3,2>` is our kernel). It outperforms some CUTLASS / cuBLASLt kernels in certain cases.

| Case: M=N=K              | Case: Large M with N=K=256               |
|:--------------------------------:|:---------------------------------:|
| ![](img/benchmark_results_m=n=k.png)      | ![](img/benchmark_results.png)   |

## Compilation

For the benchmark, first compile `hgemm.cu` by running `make`. Then you may run `python3 benchmark.py` to execute the benchmark. Time measurement is done with Nsight Compute.

## Note

The performance can still be improved by tuning the tile sizes and thread block swizzle patterns.