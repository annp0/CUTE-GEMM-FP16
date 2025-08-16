#include <stdio.h>
#include <iostream>
#include <string>
#include <cstdlib>
#include <cuda.h>

#include <cute/layout.hpp>
#include <cute/tensor.hpp>

#include "cublaslt_hgemm.cuh"

void fill_rand(cute::half_t* ptr, int size) {
  for (int i = 0; i < size; i++){
    *(ptr ++) = ((rand() % 200) - 100.f) * 0.01f;
  }
}

template< 
          const int N_STAGE = 3,
          const int N_EPILOGUE_BUF = 2
        >
__global__ __launch_bounds__(128)
void hgemm(
  const void* ptr_A, const void* ptr_B, void* ptr_C,
  int M, int N, int K
){
  using namespace cute;
  using T = half_t;

  static constexpr int BM = 128;
  static constexpr int BN = 128;
  static constexpr int BK = 32;

  using smem_layout_atom = decltype(
    Swizzle<1, 3, 3>{},
    make_layout(
      make_shape(_8{}, Int<BK>{}), 
      make_stride(Int<BK>{}, _1{}))
  );

  using smem_layout_a = decltype(
    tile_to_shape(smem_layout_atom{}, make_shape(Int<BM>{}, Int<BK>{}, Int<N_STAGE>{}))
  );

  using smem_layout_b = decltype(
    tile_to_shape(smem_layout_atom{}, make_shape(Int<BN>{}, Int<BK>{}, Int<N_STAGE>{}))
  );

  extern __shared__ T smem_data[];
  T* smem_A = smem_data;
  T* smem_B = smem_data + cosize(smem_layout_a{});

  int thr_idx = threadIdx.x;

  constexpr int GROUP_SIZE_M = 8;
  int blk_id = blockIdx.y * gridDim.x + blockIdx.x;
  int num_blocks_m = M / BM;
  int num_blocks_n = N / BN;
  int num_pid_in_group = GROUP_SIZE_M * num_blocks_n;
  int group_id = blk_id / num_pid_in_group;
  int first_pid_m = group_id * GROUP_SIZE_M;
  int group_size_m = min(num_blocks_m - first_pid_m, GROUP_SIZE_M);
  int blk_idx_y = first_pid_m + ((blk_id % num_pid_in_group) % group_size_m);
  int blk_idx_x = (blk_id % num_pid_in_group) / group_size_m;
  //int blk_idx_x = blockIdx.x;
  //int blk_idx_y = blockIdx.y;


  Tensor A = make_tensor(make_gmem_ptr((T*) ptr_A), make_shape(M, K), make_stride(K, _1{}));
  Tensor B = make_tensor(make_gmem_ptr((T*) ptr_B), make_shape(N, K), make_stride(K, _1{}));
  Tensor C = make_tensor(make_gmem_ptr((T*) ptr_C), make_shape(M, N), make_stride(N, _1{}));

  Tensor gA = local_tile(A, make_tile(Int<BM>{}, Int<BK>{}), make_coord(blk_idx_y, _));
  Tensor gB = local_tile(B, make_tile(Int<BN>{}, Int<BK>{}), make_coord(blk_idx_x, _));
  Tensor gC = local_tile(C, make_tile(Int<BM>{}, Int<BN>{}), make_coord(blk_idx_y, blk_idx_x));


  // (BM, BK, stage)
  auto sA = make_tensor(make_smem_ptr(smem_A), smem_layout_a{});
  auto sB = make_tensor(make_smem_ptr(smem_B), smem_layout_b{});

  // using m16n8k16.row.col
  using mma_traits = MMA_Traits<SM80_16x8x16_F16F16F16F16_TN>;
  using mma_atom_shape = mma_traits::Shape_MNK;
  using mma_atom = MMA_Atom<mma_traits>;
  static constexpr int MMA_THR_M = 2;
  static constexpr int MMA_THR_N = 2;
  static constexpr int MMA_THR_K = 1;
  static constexpr int MMA_REG_M = 1;
  static constexpr int MMA_REG_N = 2;
  static constexpr int MMA_REG_K = 1;
  // the tiled mma now solves problem of size 32x32x16
  static constexpr int MMA_TILE_M = get<0>(mma_atom_shape{}) * MMA_THR_M * MMA_REG_M;
  static constexpr int MMA_TILE_N = get<1>(mma_atom_shape{}) * MMA_THR_N * MMA_REG_N;
  static constexpr int MMA_TILE_K = get<2>(mma_atom_shape{}) * MMA_THR_K * MMA_REG_K;

  using mma_thr_tile = decltype(make_layout(
    make_shape(Int<MMA_THR_M>{}, Int<MMA_THR_N>{}, Int<MMA_THR_K>{}))
  );
  using mma_tile = decltype(
    make_tile(Int<MMA_TILE_M>{}, Int<MMA_TILE_N>{}, Int<MMA_TILE_K>{})
  );
  
  auto tiled_mma = make_tiled_mma(mma_atom{}, mma_thr_tile{}, mma_tile{});
  auto thr_mma = tiled_mma.get_slice(thr_idx);
  // (MMA, MMA_M, MMA_K)
  auto mma_rA = thr_mma.partition_fragment_A(gA(_, _, 0));
  //  ((2, 2), 8, 2) (MMA, MMA_N, MMA_K)
  auto mma_rB = thr_mma.partition_fragment_B(gB(_, _, 0));
  //  ((2, 2), 4, 8) (MMA, MMA_M, MMA_N)
  auto mma_rC = thr_mma.partition_fragment_C(gC);
  clear(mma_rC);

  using g2s_copy_atom = Copy_Atom<Copy_Traits<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>>, T>;
  // coalesced access, 128 byte transfer, each warp has 4 phases
  // writing is free of bank conflicts, before and after swizzle 
  using g2s_tiled_copy = decltype(make_tiled_copy(
    g2s_copy_atom{},
    make_layout(make_shape(_32{}, _4{}), make_stride(_4{}, _1{})),
    make_layout(make_shape(_1{}, _8{}))
  ));
  using g2s_tiled_copy_a = g2s_tiled_copy;
  using g2s_tiled_copy_b = g2s_tiled_copy;
  auto g2s_thr_copy_a = g2s_tiled_copy_a{}.get_slice(thr_idx);
  auto g2s_thr_copy_b = g2s_tiled_copy_b{}.get_slice(thr_idx);
  // ((_8, _1), _4, _1), repeat 4 times along M (CPY, CPY_M, CPY_K, (from gA idx_k))
  auto g2s_thr_copy_a_src = g2s_thr_copy_a.partition_S(gA);
  // same dimension as a (CPY, CPY_N, CPY_K idx_k)
  auto g2s_thr_copy_b_src = g2s_thr_copy_b.partition_S(gB);
  // (CPY, CPY_M, CPY_K, #stages)
  auto g2s_thr_copy_a_dst = g2s_thr_copy_a.partition_D(sA);
  // smem B (128, 32, #stages) -> (CPY, CPY_M, CPY_K, #stages)
  auto g2s_thr_copy_b_dst = g2s_thr_copy_b.partition_D(sB);

  // we use atom ldmatrix.x4
  // and all the 8-element-segments in the same logical column are now distrubuted across all banks
  using s2r_copy_atom = Copy_Atom<Copy_Traits<SM75_U32x4_LDSM_N>, T>;
  using s2r_tiled_copy_a = decltype(make_tiled_copy_A(s2r_copy_atom{}, tiled_mma));
  using s2r_tiled_copy_b = decltype(make_tiled_copy_B(s2r_copy_atom{}, tiled_mma));
  // now tiled copy copies 32x16 into tiled mma
  auto s2r_thr_copy_a = s2r_tiled_copy_a{}.get_slice(thr_idx);
  auto s2r_thr_copy_b = s2r_tiled_copy_b{}.get_slice(thr_idx);
  // repeat tiled copy 4 times on M, 2 times on N
  auto s2r_thr_copy_a_src = s2r_thr_copy_a.partition_S(sA);
  auto s2r_thr_copy_a_dst_view = s2r_thr_copy_a.retile_D(mma_rA);
  auto s2r_thr_copy_b_src = s2r_thr_copy_b.partition_S(sB);
  auto s2r_thr_copy_b_dst_view = s2r_thr_copy_b.retile_D(mma_rB);

  // multi stage (load -> compute) pipeline
  int idx_read_k = 0;
  int idx_write_buf = 0;
  int idx_read_buf = 0;

  #pragma unroll
  for (int idx_stage = 0; idx_stage < N_STAGE - 1; idx_stage++){
    copy(g2s_tiled_copy_a{}, g2s_thr_copy_a_src(_, _, _, idx_read_k), 
          g2s_thr_copy_a_dst(_, _, _, idx_write_buf));
    copy(g2s_tiled_copy_b{}, g2s_thr_copy_b_src(_, _, _, idx_read_k),
          g2s_thr_copy_b_dst(_, _, _, idx_write_buf));
    // commit group
    cp_async_fence();
    
    idx_read_k ++;
    idx_write_buf ++;
  }

  cp_async_wait<N_STAGE - 2>();
  __syncthreads();

  // we need to loop over MMA_K ourselves
  copy(s2r_tiled_copy_a{}, s2r_thr_copy_a_src(_, _, 0, idx_read_buf), 
          s2r_thr_copy_a_dst_view(_, _, 0));
  copy(s2r_tiled_copy_b{}, s2r_thr_copy_b_src(_, _, 0, idx_read_buf), 
          s2r_thr_copy_b_dst_view(_, _, 0));
  
  int N_BLK = K / BK;
  
  for (int idx_blk = 0; idx_blk < N_BLK; idx_blk++){
    int MMA_K = size<2>(mma_rA);

    if (idx_read_k < N_BLK){
      copy(g2s_tiled_copy_a{}, g2s_thr_copy_a_src(_, _, _, idx_read_k),
            g2s_thr_copy_a_dst(_, _, _, idx_write_buf));
      copy(g2s_tiled_copy_b{}, g2s_thr_copy_b_src(_, _, _, idx_read_k),
            g2s_thr_copy_b_dst(_, _, _, idx_write_buf));
      
      idx_read_k ++;
      idx_write_buf = (idx_write_buf + 1) % N_STAGE;
    }

    // we commit empty groups at the end so what we do not need 
    // a special case for waiting groups
    cp_async_fence();

    #pragma unroll
    for (int idx_mma_k = 0; idx_mma_k < MMA_K - 1; idx_mma_k++){
      copy(s2r_tiled_copy_a{}, s2r_thr_copy_a_src(_, _, idx_mma_k + 1, idx_read_buf), 
            s2r_thr_copy_a_dst_view(_, _, idx_mma_k + 1));
      copy(s2r_tiled_copy_b{}, s2r_thr_copy_b_src(_, _, idx_mma_k + 1, idx_read_buf), 
            s2r_thr_copy_b_dst_view(_, _, idx_mma_k + 1));
      gemm(tiled_mma, mma_rC, mma_rA(_, _, idx_mma_k), mma_rB(_, _, idx_mma_k), mma_rC);
    }

    cp_async_wait<N_STAGE - 2>();
    __syncthreads();
    // increase the index for the buffer to read, as we need to load in reg 
    // for mma_k(0) (the start) for the next block tile 
    idx_read_buf = (idx_read_buf + 1) % N_STAGE;

    // at the last tile and last mma_k we will load in trash but we do not compute on it
    copy(s2r_tiled_copy_a{}, s2r_thr_copy_a_src(_, _, 0, idx_read_buf), 
          s2r_thr_copy_a_dst_view(_, _, 0));
    copy(s2r_tiled_copy_b{}, s2r_thr_copy_b_src(_, _, 0, idx_read_buf), 
          s2r_thr_copy_b_dst_view(_, _, 0));
    gemm(tiled_mma, mma_rC, mma_rA(_, _, MMA_K - 1), mma_rB(_, _, MMA_K - 1), mma_rC);
    
  }

  // epilogue
  using smem_layout_atom_c = decltype(
    composition(Swizzle<1, 3, 3>{}, 
    make_layout(
      make_shape(Int<MMA_TILE_M>{}, Int<MMA_TILE_N>{}), 
      make_stride(Int<MMA_TILE_N>{}, _1{})
    )));
  
  using smem_layout_c = decltype(tile_to_shape(
    smem_layout_atom_c{},
    make_shape(Int<MMA_TILE_M>{}, Int<MMA_TILE_N>{}, Int<N_EPILOGUE_BUF>{})
  ));

  static_assert(size<0>(smem_layout_a{}) * size<1>(smem_layout_a{}) >= size(smem_layout_c{}));

  // reuse some smem
  auto sC = make_tensor(sA(_, _, 0).data(), smem_layout_c{});

  using r2s_copy_atom_c = Copy_Atom<UniversalCopy<int>, T>;
  using s2g_copy_atom_c = Copy_Atom<UniversalCopy<uint128_t>, T>;

  // copies 32x32
  using r2s_tiled_copy_c = decltype(make_tiled_copy_C(r2s_copy_atom_c{}, tiled_mma)); 

  // copies 32x32
  using s2g_tiled_copy_c = decltype(
    make_tiled_copy(
      s2g_copy_atom_c{},
      make_layout(make_shape(_32{}, _4{}), make_stride(_4{}, _1{})),
      make_layout(make_shape(_1{},_8{}))
    ));

  auto r2s_thr_copy_c = r2s_tiled_copy_c{}.get_slice(thr_idx);
  auto r2s_thr_copy_c_src = r2s_thr_copy_c.retile_S(mma_rC);
  // (CPY (8), _1, _1, N_BUF)
  auto r2s_thr_copy_c_dst = r2s_thr_copy_c.partition_D(sC);

  auto s2g_thr_copy_c = s2g_tiled_copy_c{}.get_slice(thr_idx);
  // (CPY, _1, _1, N_BUF)
  auto s2g_thr_copy_c_src = s2g_thr_copy_c.partition_S(sC);
  // (CPY, CPY_M, CPY_N)
  auto s2g_thr_copy_c_dst = s2g_thr_copy_c.partition_D(gC);

  auto r2s_src_grouped = group_modes<1, 3>(r2s_thr_copy_c_src);
  auto s2g_dst_grouped = group_modes<1, 3>(s2g_thr_copy_c_dst);

  #pragma unroll
  for (int idx_mn = 0; idx_mn < size<1>(r2s_src_grouped); idx_mn += N_EPILOGUE_BUF){
    
    #pragma unroll
    for (int idx_buf = 0; idx_buf < N_EPILOGUE_BUF; idx_buf ++){
      copy(r2s_tiled_copy_c{}, r2s_src_grouped(_, idx_mn + idx_buf), 
          r2s_thr_copy_c_dst(_, 0, 0, idx_buf));
    }
    __syncthreads();

    #pragma unroll
    for (int idx_buf = 0; idx_buf < N_EPILOGUE_BUF; idx_buf ++){
      copy(s2g_tiled_copy_c{}, s2g_thr_copy_c_src(_, 0, 0, idx_buf), 
          s2g_dst_grouped(_, idx_mn + idx_buf));
    }
    __syncthreads();

  }
}

int main(int argc, char* argv[]){

  using T = cute::half_t;

  srand(42);

  if (argc == 1) {
    std::cout << "Error: No arguments provided." << std::endl;
    std::cout << "Usage: " << argv[0] << " [-M=value] [-N=value] [-K=value]" << std::endl;
    return 1;
  }
  
  int M = 0, N = 0, K = 0;
  
  for (int i = 1; i < argc; i++) {
    std::string arg(argv[i]);
    if (arg.find("-M=") == 0) {
      M = std::stoi(arg.substr(3));
    } else if (arg.find("-N=") == 0) {
      N = std::stoi(arg.substr(3));
    } else if (arg.find("-K=") == 0) {
      K = std::stoi(arg.substr(3));
    } else {
      std::cout << "Unknown argument: " << arg << std::endl;
      std::cout << "Usage: " << argv[0] << " [-M=value] [-N=value] [-K=value]" << std::endl;
      return 1;
    }
  }

  if (M % 128 != 0 || N % 128 != 0 || K % 32 != 0) {
    std::cout << "Error: M, N must be divisible by 128, K must be divisible by 32" << std::endl;
    return 1;
  }
  
  std::cout << "M=" << M << ", N=" << N << ", K=" << K << std::endl;
  
  T *ptr_A;
  T *ptr_B;
  T *ptr_C;
  T *ptr_C_cblt;

  T *h_ptr_A;
  T *h_ptr_B;
  T *h_ptr_C;
  T *h_ptr_C_cblt;

  h_ptr_A = (T *)malloc(sizeof(T) * M * K);
  h_ptr_B = (T *)malloc(sizeof(T) * N * K);
  h_ptr_C = (T *)malloc(sizeof(T) * M * N);
  h_ptr_C_cblt = (T *)malloc(sizeof(T) * M * N);

  cudaMalloc(&ptr_A, sizeof(T) * M * K);
  cudaMalloc(&ptr_B, sizeof(T) * N * K);
  cudaMalloc(&ptr_C, sizeof(T) * M * N);
  cudaMalloc(&ptr_C_cblt, sizeof(T) * M * N);

  fill_rand(h_ptr_A, M * K);
  fill_rand(h_ptr_B, N * K);

  cudaMemcpy(ptr_A, h_ptr_A, sizeof(T) * M * K, cudaMemcpyHostToDevice);
  cudaMemcpy(ptr_B, h_ptr_B, sizeof(T) * N * K, cudaMemcpyHostToDevice);

  cudaMemset(ptr_C, 0, sizeof(T) * M * N);
  cudaMemset(ptr_C_cblt, 0, sizeof(T) * M * N);

  auto cblt = cublaslt_hgemm(ptr_B, ptr_A, ptr_C_cblt, M, N, K);
  cblt.do_hgemm();

  static constexpr int stages = 3;
  static constexpr int n_epi_buf = 2;
  static constexpr int smem_size = 128 * 32 * 2 * stages * 2;
  
  dim3 block(128);
  dim3 grid(N / 128, M / 128);

  std::cout << "Running my hgemm with " << stages << " stages, "
    << n_epi_buf << " epilogue buffers..." << std::endl;

  hgemm<stages, n_epi_buf><<<grid, block, smem_size>>>(
    ptr_A, ptr_B, ptr_C, 
    M, N, K
  );

  cudaMemcpy(h_ptr_C, ptr_C, sizeof(T) * M * N, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_ptr_C_cblt, ptr_C_cblt, sizeof(T) * M * N, cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();
  auto err = cudaGetLastError();

  if (err == cudaSuccess) {
    return 0;
  } else {
    printf("Cuda Error (%d): %s\n", err, cudaGetErrorString(err));
    return 1;
  }
}
