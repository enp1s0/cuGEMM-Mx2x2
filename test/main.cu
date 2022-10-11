#include <chrono>
#include <random>
#include <iostream>
#include <cugemm_Mx2x2.hpp>

constexpr unsigned num_perf_test = 1;

enum gemm_mode_t {
	none,
	sgemm,
	cgemm
};

int main(int argc, char** argv) {
	if (argc <= 4) {
		std::fprintf(stderr, "Usage: %s [GEMM mode (sgemm/cgemm)] [A Layout (N/T)] [B Layout (N/T)] [M]\n", argv[0]);
		return 1;
	}

	const std::string gemm_mode_str = argv[1];
	auto gemm_mode = gemm_mode_t::none;
	if (gemm_mode_str == "sgemm") {
		gemm_mode = gemm_mode_t::sgemm;
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

	auto mat_a_size = M * 2lu;
	auto mat_b_size = 2 * 2lu;
	auto mat_c_size = M * 2lu;

	std::size_t complexity = 2lu * 2 * 2 * M;
	std::size_t num_elements = 2lu * 2lu + 2lu * M + 2lu * M;
	if (gemm_mode == gemm_mode_t::sgemm) {
		mat_a_size *= sizeof(float);
		mat_b_size *= sizeof(float);
		mat_c_size *= sizeof(float);
		num_elements *= sizeof(float);
	} else if (gemm_mode == gemm_mode_t::cgemm) {
		mat_a_size *= 2 * sizeof(float);
		mat_b_size *= 2 * sizeof(float);
		mat_c_size *= 2 * sizeof(float);
		num_elements *= 2 * sizeof(float);
		complexity *= 2lu;
	}

	float *host_mat_a, *host_mat_b, *host_mat_c;
	cudaMallocHost(&host_mat_a, mat_a_size);
	cudaMallocHost(&host_mat_b, mat_a_size);
	cudaMallocHost(&host_mat_c, mat_a_size);
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

	if (gemm_mode == gemm_mode_t::sgemm) {
		const auto alpha = 1.f;
		const auto beta = 0.f;
		mtk::cugemm::gemm_Mx2x2(
				op_a, op_b,
				M,
				alpha,
				dev_mat_a, (op_a == CUBLAS_OP_N ? M : 2),
				dev_mat_b, 2,
				beta,
				dev_mat_c, M
				);
		// throughput
		cudaDeviceSynchronize();
		const auto start_clock = std::chrono::system_clock::now();
		for (unsigned i = 0; i < num_perf_test; i++) {
			mtk::cugemm::gemm_Mx2x2(
					op_a, op_b,
					M,
					alpha,
					dev_mat_a, (op_a == CUBLAS_OP_N ? M : 2),
					dev_mat_b, 2,
					beta,
					dev_mat_c, M
					);
		}
		cudaDeviceSynchronize();
		const auto end_clock = std::chrono::system_clock::now();
		elapsed_time_per_gemm = std::chrono::duration_cast<std::chrono::nanoseconds>(end_clock - start_clock).count() * 1e-9 / num_perf_test;
	} else if (gemm_mode == gemm_mode_t::cgemm) {
		const auto alpha = make_cuComplex(1.f, 0.f);
		const auto beta = make_cuComplex(0.f, 0.f);
		mtk::cugemm::gemm_Mx2x2(
				op_a, op_b,
				M,
				alpha,
				reinterpret_cast<cuComplex*>(dev_mat_a), (op_a == CUBLAS_OP_N ? M : 2),
				reinterpret_cast<cuComplex*>(dev_mat_b), 2,
				beta,
				reinterpret_cast<cuComplex*>(dev_mat_c), M
				);
		// throughput
		cudaDeviceSynchronize();
		const auto start_clock = std::chrono::system_clock::now();
		for (unsigned i = 0; i < num_perf_test; i++) {
			mtk::cugemm::gemm_Mx2x2(
					op_a, op_b,
					M,
					alpha,
					reinterpret_cast<cuComplex*>(dev_mat_a), (op_a == CUBLAS_OP_N ? M : 2),
					reinterpret_cast<cuComplex*>(dev_mat_b), 2,
					beta,
					reinterpret_cast<cuComplex*>(dev_mat_c), M
					);
		}
		cudaDeviceSynchronize();
		const auto end_clock = std::chrono::system_clock::now();
		elapsed_time_per_gemm = std::chrono::duration_cast<std::chrono::nanoseconds>(end_clock - start_clock).count() * 1e-9 / num_perf_test;
	}

	const auto throughput_in_tflops = complexity / elapsed_time_per_gemm * 1e-12;
	const auto bandwidth_in_tb_per_s = num_elements / elapsed_time_per_gemm * 1e-12;

	std::printf("%lu,%e,%e\n",
			M,
			throughput_in_tflops,
			bandwidth_in_tb_per_s
			);

	cudaFree(dev_mat_a);
	cudaFree(dev_mat_b);
	cudaFree(dev_mat_c);

	cudaFreeHost(host_mat_a);
	cudaFreeHost(host_mat_b);
	cudaFreeHost(host_mat_c);
}
