#include <cublasLt.h>
#include <iostream>

#include <cute/layout.hpp>

#define cublas_check(cmd) do { \
    cublasStatus_t s = cmd; \
    if (s != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "CUBLAS Error: %d at %s:%d\n", s, __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// A: K x M, col major, B: K x N, col major, C: M x N, col major
// C = AT B
class cublaslt_hgemm {
public:
  cublaslt_hgemm(const cute::half_t* ptr_A, 
                 const cute::half_t* ptr_B, 
                 cute::half_t* ptr_C, 
                 int M, int N, int K);
  ~cublaslt_hgemm();
  cublaslt_hgemm(const cublaslt_hgemm&) = delete;
  cublaslt_hgemm& operator=(const cublaslt_hgemm&) = delete;
  void do_hgemm();

private:
  cublasLtHandle_t handle_ = nullptr;
  cublasLtMatrixLayout_t A_desc_ = nullptr;
  cublasLtMatrixLayout_t B_desc_ = nullptr;
  cublasLtMatrixLayout_t C_desc_ = nullptr;
  cublasLtMatmulDesc_t matmul_desc_ = nullptr;
  cublasLtMatmulPreference_t preference_ = nullptr;

  static constexpr int max_num_algo_ = 1024;
  cublasLtMatmulHeuristicResult_t algo_list_[max_num_algo_]{};
  int num_algo_ = 0;

  cute::half_t alpha_ = cute::half_t(1.f);
  cute::half_t beta_ = cute::half_t(0.f);

  void* workspace_ = nullptr;
  int workspace_size_ = 0;

  const cute::half_t* A_ = nullptr;
  const cute::half_t* B_ = nullptr;
  cute::half_t* C_ = nullptr;
};

cublaslt_hgemm::cublaslt_hgemm(
  const cute::half_t* ptr_A, 
  const cute::half_t* ptr_B, 
  cute::half_t* ptr_C, 
  int M, int N, int K
) : A_(ptr_A), B_(ptr_B), C_(ptr_C){

  auto version = cublasLtGetVersion();
  std::cout << "cublasLt version " << version << std::endl;

  cublas_check(cublasLtCreate(&handle_));

  cublasOperation_t A_trans = CUBLAS_OP_T;
  cublasOperation_t B_trans = CUBLAS_OP_N;

  cublas_check(cublasLtMatrixLayoutCreate(&A_desc_, CUDA_R_16F, K, M, K));
  cublas_check(cublasLtMatrixLayoutCreate(&B_desc_, CUDA_R_16F, K, N, K));
  cublas_check(cublasLtMatrixLayoutCreate(&C_desc_, CUDA_R_16F, M, N, M));

  cublas_check(cublasLtMatmulDescCreate(&matmul_desc_, CUBLAS_COMPUTE_16F, CUDA_R_16F));
  cublas_check(cublasLtMatmulDescSetAttribute(matmul_desc_, 
    CUBLASLT_MATMUL_DESC_TRANSA, &A_trans, sizeof(A_trans)));
  cublas_check(cublasLtMatmulDescSetAttribute(matmul_desc_, 
    CUBLASLT_MATMUL_DESC_TRANSB, &B_trans, sizeof(B_trans)));

  cublasLtMatmulPreferenceCreate(&preference_);

  // rule based, will not launch kernels
  cublasLtMatmulAlgoGetHeuristic(
    handle_, matmul_desc_, A_desc_, B_desc_,
    C_desc_, C_desc_, preference_, max_num_algo_,
    algo_list_, &num_algo_);
}

void cublaslt_hgemm::do_hgemm() {
  std::cout << "Running cublasLt with " << num_algo_ << " algorithms..." << std::endl;
  for (int i = 0; i < num_algo_; i ++){
    auto algo = algo_list_[i];
    cublas_check(cublasLtMatmul(
      handle_, matmul_desc_, &alpha_, 
      A_, A_desc_, B_, B_desc_, &beta_, 
      C_, C_desc_, C_, C_desc_,
      &(algo.algo), workspace_, workspace_size_, 
      0)
    );
  }
}

cublaslt_hgemm::~cublaslt_hgemm() {
    if (workspace_) {
        cudaFree(workspace_);
        workspace_ = nullptr;
    }

    if (preference_) {
        cublas_check(cublasLtMatmulPreferenceDestroy(preference_));
        preference_ = nullptr;
    }

    if (matmul_desc_) {
        cublas_check(cublasLtMatmulDescDestroy(matmul_desc_));
        matmul_desc_ = nullptr;
    }

    if (A_desc_) {
        cublas_check(cublasLtMatrixLayoutDestroy(A_desc_));
        A_desc_ = nullptr;
    }

    if (B_desc_) {
        cublas_check(cublasLtMatrixLayoutDestroy(B_desc_));
        B_desc_ = nullptr;
    }

    if (C_desc_) {
        cublas_check(cublasLtMatrixLayoutDestroy(C_desc_));
        C_desc_ = nullptr;
    }

    if (handle_) {
        cublas_check(cublasLtDestroy(handle_));
        handle_ = nullptr;
    }
}