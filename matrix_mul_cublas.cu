#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

void initWith(float* M, int dim, float n) {
	for (int i = 0; i < dim; i++) {
		for (int j = 0; j < dim; j++) {
			M[i * dim + j] = n;
		}
	}
}

void init(float* M, int dim) {
	for (int i = 0; i < dim; i++) {
		for (int j = 0; j < dim; j++) {
			M[i * dim + j] = rand();
		}
	}
}

void checkResult(float* A, float* B, float* C, float* C_cpu, int dim) {
	for (int i = 0; i < dim; i++) {
		for (int j = 0; j < dim; j++) {
			if (abs(C[i * dim + j] - C_cpu[i * dim + j]) > 0.001) {
				printf("matrix pos: %d,%d\n", i, j);
				printf("index: %d\n", i * dim + j); // DEBUG PRINT
				printf("CPU: %f, GPU %f\n", C_cpu[i * dim + j], C[i * dim + j]);	 // DEBUG PRINT
				printf("ERROR: Incorrect Results! %f\n", abs(C_cpu[i * dim + j] - C[i * dim + j]));
				return;
			}
		}
	}

	printf("Everything is OK! :D\n");
}

int main(int argc, char** argv) {
	srand(time(0));
	cudaError_t error;
	cublasStatus_t status;
	cublasHandle_t handle;

	int N = 100;
	size_t size = N * N * sizeof(float);

	float* h_A, *h_B, *h_C, *h_C_cpu;

	h_A = (float*) malloc(size);
	h_B = (float*) malloc(size);
	h_C = (float*) malloc(size);
	h_C_cpu = (float*) malloc(size);

	initWith(h_A, N, 1.0f);
	initWith(h_B, N, 1.0f);
	initWith(h_C_cpu, N, 100.0f);

	float* d_A, *d_B, *d_C;

	error = cudaMalloc((void**) &d_A, size);
	error = cudaMalloc((void**) &d_B, size);
	error = cudaMalloc((void**) &d_C, size);

	status = cublasCreate(&handle);

	float al = 1.0f, bet = 0.0f;

	status = cublasSetMatrix(N, N, size, h_A, N, d_A, N);
	status = cublasSetMatrix(N, N, size, h_B, N, d_B, N);
	status = cublasSetMatrix(N, N, size, h_C, N, d_C, N);


	status  = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &al, d_A, N, d_B, N, &bet, d_C, N);

	status = cublasGetMatrix(N, N, sizeof(float), d_C, N, h_C, N);

	printf("Before checkResult	\n"); // DEBUG PRINT
	checkResult(h_A, h_B, h_C, h_C_cpu, N);
	printf("After checkResult\n"); // DEBUG PRINT
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	cublasDestroy(handle);

	free(h_A);
	free(h_B);
	free(h_C);

	return 0;
}
