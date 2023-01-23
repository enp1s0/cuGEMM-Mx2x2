#pragma once
#include <cstdint>
#include <cublas_v2.h>

namespace mtk {
namespace cugemm {
template <class T>
void gemm_Mx2x2(
		const cublasOperation_t op_a,
		const cublasOperation_t op_b,
		const unsigned M,
		const T alpha,
		const T* const a_ptr, const std::size_t lda,
		const T* const b_ptr, const std::size_t ldb,
		const T beta,
		T* const c_ptr, const std::size_t ldc,
		cudaStream_t cuda_stream = 0
		);
template <class T>
void gemm_strided_batch_Mx2x2(
		const cublasOperation_t op_a,
		const cublasOperation_t op_b,
		const unsigned M,
		const T alpha,
		const T* const a_ptr, const std::size_t lda, const std::size_t stridea,
		const T* const b_ptr, const std::size_t ldb, const std::size_t strideb,
		const T beta,
		T* const c_ptr, const std::size_t ldc, const std::size_t stridec,
		const unsigned batch_count,
		cudaStream_t cuda_stream = 0
		);
template <class T>
void gemm_2xNx2(
		const cublasOperation_t op_a,
		const cublasOperation_t op_b,
		const unsigned N,
		const T alpha,
		const T* const a_ptr, const std::size_t lda,
		const T* const b_ptr, const std::size_t ldb,
		const T beta,
		T* const c_ptr, const std::size_t ldc,
		cudaStream_t cuda_stream = 0
		);
template <class T>
void gemm_strided_batch_2xNx2(
		const cublasOperation_t op_a,
		const cublasOperation_t op_b,
		const unsigned N,
		const T alpha,
		const T* const a_ptr, const std::size_t lda, const std::size_t stridea,
		const T* const b_ptr, const std::size_t ldb, const std::size_t strideb,
		const T beta,
		T* const c_ptr, const std::size_t ldc, const std::size_t stridec,
		const unsigned batch_count,
		cudaStream_t cuda_stream = 0
		);
} // namespace cugemm
} // namespace mtk

