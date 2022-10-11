#include <type_traits>
#include <algorithm>
#include <cassert>
#include <cuComplex.h>
#include <cugemm_Mx2x2.hpp>

struct col_major;
struct row_major;

namespace detail {
bool is_zero(const float a) {return a == 0;}
bool is_zero(const cuComplex a) {return is_zero(a.x) && is_zero(a.y);}

__device__ float fma(const float a, const float b, const float c) {return fmaf(a, b, c);}
__device__ cuComplex fma(const cuComplex a, const cuComplex b, const cuComplex c) {
	return make_cuComplex(
			fma(a.x, b.x, fma(-a.y, b.y, c.x)),
			fma(a.y, b.x, fma( a.y, b.x, c.y))
			);
}

__device__ float mul(const float a, const float b) {return mul(a, b);}
__device__ cuComplex mul(const cuComplex a, const cuComplex b) {
	return make_cuComplex(
			fma(a.x, b.x, mul(-a.y, b.y)),
			fma(a.y, b.x, mul( a.y, b.x))
			);
}

template <class T>
__device__ T zero() {return 0.f;}
template <>
__device__ cuComplex zero<cuComplex>() {return make_cuComplex(0.f, 0.f);}
} // namespace detail

namespace {
template <class T, class LAYOUT_A, class LAYOUT_B, unsigned BLOCK_SIZE, unsigned M_PER_THREAD, bool BETA, unsigned N = 2, unsigned K = 2>
__device__ void gemm_core(
		const unsigned M,
		const T alpha,
		const T* const a_ptr, const std::size_t lda,
		const T* const b_ptr, const std::size_t ldb,
		const T beta,
		T* const c_ptr, const std::size_t ldc
		) {
	constexpr unsigned NUM_STAGES = 2;
	T frag_b[N * K];
	T frag_a[K * M_PER_THREAD * NUM_STAGES];
	T frag_c[N * M_PER_THREAD * NUM_STAGES];

	// Load B
	for (unsigned n = 0; n < N; n++) {
		for (unsigned k = 0; k < K; k++) {
			std::size_t index;
			if (std::is_same<LAYOUT_B, col_major>::value) {
				index = k + n * ldb;
			} else {
				index = n + k * ldb;
			}
			frag_b[k + n * K] = a_ptr[index];
		}
	}
	unsigned m_offset = M_PER_THREAD * (threadIdx.x + blockIdx.x * blockDim.x);
	if (m_offset >= M) {
		return;
	}

	// Load A
	for (unsigned m = 0; m < M_PER_THREAD; m++) {
		for (unsigned k = 0; k < K; k++) {
			std::size_t index;
			if (std::is_same<LAYOUT_A, col_major>::value) {
				index = m + k * lda;
			} else {
				index = k + m * lda;
			}
			frag_a[k + m * K] = a_ptr[index];
		}
	}

	// Load C
	if (!BETA) {
		for (unsigned m = 0; m < M_PER_THREAD; m++) {
			for (unsigned n = 0; n < N; n++) {
				std::size_t index = m + n * ldc;
				frag_c[n + m * N] = c_ptr[index];
			}
		}
	}

	//
	m_offset += M_PER_THREAD * gridDim.x * blockDim.x;
	unsigned stage = 1;
	for (; m_offset < M; m_offset += M_PER_THREAD * gridDim.x * blockDim.x) {
		// Load A
		for (unsigned m = 0; m < M_PER_THREAD; m++) {
			for (unsigned k = 0; k < K; k++) {
				std::size_t index;
				if (std::is_same<LAYOUT_A, col_major>::value) {
					index = (m + m_offset) + k * lda;
				} else {
					index = k + (m + m_offset) * lda;
				}
				frag_a[k + m * K + stage * M_PER_THREAD * K] = a_ptr[index];
			}
		}

		// Load C
		if (!BETA) {
			for (unsigned m = 0; m < M_PER_THREAD; m++) {
				for (unsigned n = 0; n < N; n++) {
					std::size_t index = m + m_offset + n * ldc;
					frag_c[n + m * N + stage * M_PER_THREAD * N] = c_ptr[index];
				}
			}
		}

		stage = 1 - stage;
		for (unsigned m = 0; m < M_PER_THREAD; m++) {
			for (unsigned n = 0; n < N; n++) {
				auto c = detail::zero<T>();
				for (unsigned k = 0; k < K; k++) {
					c = detail::fma(frag_a[m * K + k + stage * M_PER_THREAD * N], frag_b[n * K + k + stage * M_PER_THREAD * N], c);
				}
				if (BETA) {
					c = detail::fma(alpha, c, detail::mul(beta, frag_c[m + n * M_PER_THREAD + stage * M_PER_THREAD * N]));
				} else {
					c = detail::mul(alpha, c);
				}

				std::size_t index = m + m_offset + n * ldc - M_PER_THREAD * gridDim.x * blockDim.x;
				c_ptr[index] = c;
			}
		}
	}
	stage = 1 - stage;
	for (unsigned m = 0; m < M_PER_THREAD; m++) {
		for (unsigned n = 0; n < N; n++) {
			auto c = detail::zero<T>();
			for (unsigned k = 0; k < K; k++) {
				c = detail::fma(frag_a[m * K + k + stage * M_PER_THREAD * N], frag_b[n * K + k + stage * M_PER_THREAD * N], c);
			}
			if (BETA) {
				c = detail::fma(alpha, c, detail::mul(beta, frag_c[m + n * M_PER_THREAD + stage * M_PER_THREAD * N]));
			} else {
				c = detail::mul(alpha, c);
			}

			std::size_t index = m + m_offset + n * ldc - M_PER_THREAD * gridDim.x * blockDim.x;
			c_ptr[index] = c;
		}
	}
}

template <class T, class LAYOUT_A, class LAYOUT_B, unsigned BLOCK_SIZE, unsigned M_PER_THREAD, bool BETA, unsigned N = 2, unsigned K = 2>
__global__ void gemm_kernel(
		const unsigned M,
		const T alpha,
		const T* const a_ptr, const std::size_t lda,
		const T* const b_ptr, const std::size_t ldb,
		const T beta,
		T* const c_ptr, const std::size_t ldc
		) {
	gemm_core<T, LAYOUT_A, LAYOUT_B, BLOCK_SIZE, M_PER_THREAD, BETA, N, K>(M, alpha, a_ptr, lda, b_ptr, ldb, beta, c_ptr, ldc);
}

template <class T, class LAYOUT_A, class LAYOUT_B, unsigned BLOCK_SIZE, unsigned M_PER_THREAD, unsigned N = 2, unsigned K = 2>
void gemm_internal(
		const unsigned M,
		const T alpha,
		const T* const a_ptr, const std::size_t lda,
		const T* const b_ptr, const std::size_t ldb,
		const T beta,
		T* const c_ptr, const std::size_t ldc
		) {
	assert(M >= M_PER_THREAD);
	assert((M & (M - 1)) == 0);
	const auto min_grid_size = std::max<unsigned>(M / (M_PER_THREAD * BLOCK_SIZE), 1u);
	const auto grid_size = std::min<unsigned>(min_grid_size, 1024u);

	if (detail::is_zero(beta)) {
		gemm_kernel<T, LAYOUT_A, LAYOUT_B, BLOCK_SIZE, M_PER_THREAD, false, N, K><<<grid_size, BLOCK_SIZE>>>(M, alpha, a_ptr, lda, b_ptr, ldb, beta, c_ptr, ldc);
	} else {
		gemm_kernel<T, LAYOUT_A, LAYOUT_B, BLOCK_SIZE, M_PER_THREAD, true , N, K><<<grid_size, BLOCK_SIZE>>>(M, alpha, a_ptr, lda, b_ptr, ldb, beta, c_ptr, ldc);
	}
}
} // unnamed namespace

template <>
void mtk::cugemm::gemm_Mx2x2<float>(
		const cublasOperation_t op_a,
		const cublasOperation_t op_b,
		const unsigned M,
		const float alpha,
		const float* const a_ptr, const std::size_t lda,
		const float* const b_ptr, const std::size_t ldb,
		const float beta,
		float* const c_ptr, const std::size_t ldc
		) {
	if (op_a == CUBLAS_OP_N && op_b == CUBLAS_OP_N) {
		constexpr unsigned M_PER_THREAD = 4;
		constexpr unsigned BLOCK_SIZE = 256;
		gemm_internal<float, col_major, col_major, BLOCK_SIZE, M_PER_THREAD>(M, alpha, a_ptr, lda, b_ptr, ldb, beta, c_ptr, ldc);
	}
}

template <>
void mtk::cugemm::gemm_Mx2x2<cuComplex>(
		const cublasOperation_t op_a,
		const cublasOperation_t op_b,
		const unsigned M,
		const cuComplex alpha,
		const cuComplex* const a_ptr, const std::size_t lda,
		const cuComplex* const b_ptr, const std::size_t ldb,
		const cuComplex beta,
		cuComplex* const c_ptr, const std::size_t ldc
		) {
	if (op_a == CUBLAS_OP_N && op_b == CUBLAS_OP_N) {
		constexpr unsigned M_PER_THREAD = 4;
		constexpr unsigned BLOCK_SIZE = 256;
		gemm_internal<cuComplex, col_major, col_major, BLOCK_SIZE, M_PER_THREAD>(M, alpha, a_ptr, lda, b_ptr, ldb, beta, c_ptr, ldc);
	}
}
