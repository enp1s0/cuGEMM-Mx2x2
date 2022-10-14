#include <chrono>
#include <random>
#include <iostream>
#include <cugemm_Mx2x2.hpp>

constexpr unsigned num_perf_test = 10;

namespace {
float fma(const float a, const float b, const float c) {return a * b + c;}
cuComplex fma(const cuComplex a, const cuComplex b, const cuComplex c) {return make_cuComplex(a.x * b.x - a.y * b.y + c.x, a.x * b.y + a.y * b.x + c.y);}

float mul(const float a, const float b) {return a * b;}
cuComplex mul(const cuComplex a, const cuComplex b) {return make_cuComplex(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);}

float sub(const float a, const float b) {return a - b;}
cuComplex sub(const cuComplex a, const cuComplex b) {return make_cuComplex(a.x - b.x, a.y - b.y);}

double norm2(const float a) {return a * a;};
double norm2(const cuComplex a) {return norm2(a.x) + norm2(a.y);};

bool is_zero(const float a) {return a == 0;}
bool is_zero(const cuComplex a) {return is_zero(a.x) && is_zero(a.y);}
template <class T>
T zero() {return 0;}
template <>
cuComplex zero<cuComplex>() {return make_cuComplex(0, 0);}

template <class T>
double gemm_Mx2x2_residual(
		const cublasOperation_t op_a,
		const cublasOperation_t op_b,
		const unsigned M,
		const T alpha,
		const T* const a_ptr, const std::size_t lda,
		const T* const b_ptr, const std::size_t ldb,
		const T beta,
		const T* const c_ptr, const std::size_t ldc,
		T* const t_ptr, const std::size_t ldt
		) {
	double diff_norm2 = 0.;
	double base_norm2 = 0.;
#pragma omp parallel for reduction(+: diff_norm2) reduction(+: base_norm2)
	for (unsigned m = 0; m < M; m++) {
		for (unsigned n = 0; n < 2; n++) {
			T c = zero<T>();
			for (unsigned k = 0; k < 2; k++) {
				const auto a_index = (op_a == CUBLAS_OP_N) ? (m + lda * k) : (m * lda + k);
				const auto b_index = (op_b == CUBLAS_OP_N) ? (k + ldb * n) : (k * ldb + n);

				c = fma(a_ptr[a_index], b_ptr[b_index], c);
			}
			if (is_zero(beta)) {
				c = mul(c, alpha);
			} else {
				c = fma(c, alpha, mul(c_ptr[m + n * ldc], beta));
			}
			base_norm2 += norm2(c);
			diff_norm2 += norm2(sub(c, t_ptr[m + n * ldt]));
		}
	}
	return std::sqrt(diff_norm2 / base_norm2);
}
} // unnamed namespace

enum gemm_mode_t {
	none,
	sgemm,
	cgemm,
};

int main(int argc, char** argv) {
	if (argc <= 4) {
		std::fprintf(stderr, "Usage: %s [GEMM mode (sgemm/cgemm)] [A Layout (N/T)] [B Layout (N/T)] [M] [(Optional) batch count]\n", argv[0]);
		return 1;
	}
	std::printf("mode,M,batch_size,residual,throughput_in_tflops,bw\n");

	const std::string gemm_mode_str = argv[1];
	auto gemm_mode = gemm_mode_t::none;
	if (gemm_mode_str == "sgemm") {
		gemm_mode = gemm_mode_t::sgemm;
	} else if (gemm_mode_str == "cgemm") {
		gemm_mode = gemm_mode_t::cgemm;
	}

	unsigned batch_size = 1;
	if (argc > 5) {
		batch_size = std::stoul(argv[5]);
	}


	const std::string op_a_str = argv[2];
	const std::string op_b_str = argv[3];
	cublasOperation_t op_a, op_b;
	if (op_a_str == "N") {
		op_a = CUBLAS_OP_N;
	} else if (op_a_str == "T") {
		op_a = CUBLAS_OP_T;
	}
	if (op_b_str == "N") {
		op_b = CUBLAS_OP_N;
	} else if (op_b_str == "T") {
		op_b = CUBLAS_OP_T;
	}
	const auto M = std::stoul(argv[4]);

	auto mat_a_size = M * 2lu * batch_size * sizeof(float);
	auto mat_b_size = 2 * 2lu * batch_size * sizeof(float);
	auto mat_c_size = M * 2lu * batch_size * sizeof(float);

	std::size_t complexity = 2lu * 2 * 2 * M * batch_size;
	std::size_t num_elements = (2lu * 2lu + 2lu * M + 2lu * M) * batch_size;
	if (gemm_mode == gemm_mode_t::cgemm) {
		mat_a_size *= 2;
		mat_b_size *= 2;
		mat_c_size *= 2;
		num_elements *= 2;
		complexity *= 4lu;
	}

	float *host_mat_a, *host_mat_b, *host_mat_c, *host_mat_t;
	cudaMallocHost(&host_mat_a, mat_a_size);
	cudaMallocHost(&host_mat_b, mat_a_size);
	cudaMallocHost(&host_mat_c, mat_a_size);
	cudaMallocHost(&host_mat_t, mat_a_size);
	std::mt19937 mt(0);
	if (gemm_mode == gemm_mode_t::sgemm || gemm_mode_t::cgemm) {
		std::uniform_real_distribution<float> dist(-1, 1);
		for (std::size_t i = 0; i < mat_a_size / sizeof(float); i++) {
			host_mat_a[i] = dist(mt);
		}
		for (std::size_t i = 0; i < mat_b_size / sizeof(float); i++) {
			host_mat_b[i] = dist(mt);
		}
		for (std::size_t i = 0; i < mat_c_size / sizeof(float); i++) {
			host_mat_c[i] = 0;
		}
	}

	float *dev_mat_a, *dev_mat_b, *dev_mat_c;
	cudaMalloc(&dev_mat_a, mat_a_size);
	cudaMalloc(&dev_mat_b, mat_a_size);
	cudaMalloc(&dev_mat_c, mat_a_size);
	cudaMemcpy(dev_mat_a, host_mat_a, mat_a_size, cudaMemcpyDefault);
	cudaMemcpy(dev_mat_b, host_mat_b, mat_b_size, cudaMemcpyDefault);
	cudaMemcpy(dev_mat_c, host_mat_c, mat_c_size, cudaMemcpyDefault);

	double elapsed_time_per_gemm = 0;
	double residual = 0;

	if (gemm_mode == gemm_mode_t::sgemm) {
		const auto alpha = 1.f;
		const auto beta = 0.f;
		if (batch_size == 1) {
			mtk::cugemm::gemm_Mx2x2(
					op_a, op_b,
					M,
					alpha,
					dev_mat_a, (op_a == CUBLAS_OP_N ? M : 2),
					dev_mat_b, 2,
					beta,
					dev_mat_c, M
					);
		} else {
			mtk::cugemm::gemm_strided_batch_Mx2x2(
					op_a, op_b,
					M,
					alpha,
					dev_mat_a, (op_a == CUBLAS_OP_N ? M : 2), M * 2,
					dev_mat_b, 2, 2 * 2,
					beta,
					dev_mat_c, M, M * 2,
					batch_size
					);
		}
		cudaMemcpy(host_mat_t, dev_mat_c, mat_c_size, cudaMemcpyDefault);
		for (unsigned b = 0; b < batch_size; b++) {
			residual += gemm_Mx2x2_residual(
					op_a, op_b,
					M,
					alpha,
					host_mat_a + M * 2 * b, (op_a == CUBLAS_OP_N ? M : 2),
					host_mat_b + 2 * 2 * b, 2,
					beta,
					host_mat_c + M * 2 * b, M,
					host_mat_t + M * 2 * b, M
					);
		}
		residual /= batch_size;
		// throughput
		cudaDeviceSynchronize();
		const auto start_clock = std::chrono::system_clock::now();
		for (unsigned i = 0; i < num_perf_test; i++) {
			if (batch_size == 1) {
				mtk::cugemm::gemm_Mx2x2(
						op_a, op_b,
						M,
						alpha,
						dev_mat_a, (op_a == CUBLAS_OP_N ? M : 2),
						dev_mat_b, 2,
						beta,
						dev_mat_c, M
						);
			} else {
				mtk::cugemm::gemm_strided_batch_Mx2x2(
						op_a, op_b,
						M,
						alpha,
						dev_mat_a, (op_a == CUBLAS_OP_N ? M : 2), M * 2,
						dev_mat_b, 2, 2 * 2,
						beta,
						dev_mat_c, M, M * 2,
						batch_size
						);
			}
		}
		cudaDeviceSynchronize();
		const auto end_clock = std::chrono::system_clock::now();
		elapsed_time_per_gemm = std::chrono::duration_cast<std::chrono::nanoseconds>(end_clock - start_clock).count() * 1e-9 / num_perf_test;
	} else if (gemm_mode == gemm_mode_t::cgemm) {
		const auto alpha = make_cuComplex(1.f, 0.f);
		const auto beta = make_cuComplex(0.f, 0.f);
		if (batch_size == 1) {
			mtk::cugemm::gemm_Mx2x2(
					op_a, op_b,
					M,
					alpha,
					reinterpret_cast<cuComplex*>(dev_mat_a), (op_a == CUBLAS_OP_N ? M : 2),
					reinterpret_cast<cuComplex*>(dev_mat_b), 2,
					beta,
					reinterpret_cast<cuComplex*>(dev_mat_c), M
					);
		} else {
			mtk::cugemm::gemm_strided_batch_Mx2x2(
					op_a, op_b,
					M,
					alpha,
					reinterpret_cast<cuComplex*>(dev_mat_a), (op_a == CUBLAS_OP_N ? M : 2), M * 2,
					reinterpret_cast<cuComplex*>(dev_mat_b), 2, 2 * 2,
					beta,
					reinterpret_cast<cuComplex*>(dev_mat_c), M, M * 2,
					batch_size
					);
		}
		cudaMemcpy(host_mat_t, dev_mat_c, mat_c_size, cudaMemcpyDefault);
		for (unsigned b = 0; b < batch_size; b++) {
			residual += gemm_Mx2x2_residual(
					op_a, op_b,
					M,
					alpha,
					reinterpret_cast<cuComplex*>(host_mat_a) + M * 2 * b, (op_a == CUBLAS_OP_N ? M : 2),
					reinterpret_cast<cuComplex*>(host_mat_b) + 2 * 2 * b, 2,
					beta,
					reinterpret_cast<cuComplex*>(host_mat_c) + M * 2 * b, M,
					reinterpret_cast<cuComplex*>(host_mat_t) + M * 2 * b, M
					);
		}
		residual /= batch_size;
		// throughput
		cudaDeviceSynchronize();
		const auto start_clock = std::chrono::system_clock::now();
		for (unsigned i = 0; i < num_perf_test; i++) {
			if (batch_size == 1) {
				mtk::cugemm::gemm_Mx2x2(
						op_a, op_b,
						M,
						alpha,
						reinterpret_cast<cuComplex*>(dev_mat_a), (op_a == CUBLAS_OP_N ? M : 2),
						reinterpret_cast<cuComplex*>(dev_mat_b), 2,
						beta,
						reinterpret_cast<cuComplex*>(dev_mat_c), M
						);
			} else {
				mtk::cugemm::gemm_strided_batch_Mx2x2(
						op_a, op_b,
						M,
						alpha,
						reinterpret_cast<cuComplex*>(dev_mat_a), (op_a == CUBLAS_OP_N ? M : 2), M * 2,
						reinterpret_cast<cuComplex*>(dev_mat_b), 2, 2 * 2,
						beta,
						reinterpret_cast<cuComplex*>(dev_mat_c), M, M * 2,
						batch_size
						);
			}
		}
		cudaDeviceSynchronize();
		const auto end_clock = std::chrono::system_clock::now();
		elapsed_time_per_gemm = std::chrono::duration_cast<std::chrono::nanoseconds>(end_clock - start_clock).count() * 1e-9 / num_perf_test;
	}

	const auto throughput_in_tflops = complexity / elapsed_time_per_gemm * 1e-12;
	const auto bandwidth_in_tb_per_s = num_elements * sizeof(float) / elapsed_time_per_gemm * 1e-12;

	std::printf("%s,%lu,%u,%e,%e,%e\n",
			gemm_mode_str.c_str(),
			M,
			batch_size,
			residual,
			throughput_in_tflops,
			bandwidth_in_tb_per_s
			);

	cudaFree(dev_mat_a);
	cudaFree(dev_mat_b);
	cudaFree(dev_mat_c);

	cudaFreeHost(host_mat_a);
	cudaFreeHost(host_mat_b);
	cudaFreeHost(host_mat_c);
	cudaFreeHost(host_mat_t);
}
