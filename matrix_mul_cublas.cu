#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

void initWith(float* M, int dim) {
	for (int i = 0; i < dim; i++) {
		for (int j = 0; j < dim; j++) {
			M[i * dim + j] = rand();
		}
	}
}

void checkResult(float* A, float* B, float* C, int dim) {
	float cpu_result[dim * dim];

	for (int i = 0; i < dim; i++) {
		for (int j = 0; j < dim; j++) {
			float tmp = 0.0;
			for (int k = 0; k < dim; k++) {
				tmp += A[i * dim + k] * B[k * dim + j];
			}
			cpu_result[i * dim + j]  = tmp;
		}
	}

	for (int i = 0; i < dim; i++) {
		for (int j = 0; j < dim; j++) {
			if (cpu_result[i * dim + j] != C[i * dim + j]) {
				printf("ERROR: Incorrect Results!\n");
				return;
			}
		}
	}

	printf("Everything is OK! :D\n");
}

int main(int argc, char** argv) {
	cudaError_t error;
	cublasStatus_t status;
	cublasHandle_t handle;

	int N = 100;
	size_t size = N * N * sizeof(float);

	float* h_A, *h_B, *h_C;

	h_A = (float*) malloc(size);
	h_B = (float*) malloc(size);
	h_B = (float*) malloc(size);

	initWith(h_A, N);
	initWith(h_B, N);

	float* d_A, *d_B, *d_C;

	error = cudaMalloc((void**) &d_A, size);
	error = cudaMalloc((void**) &d_B, size);
	error = cudaMalloc((void**) &d_C, size);

	status = cublasCreate(&handle);

	float al = 1.0f, bet = 1.0f;

	status = cublasSetMatrix(N, N, size, h_A, N, d_A, N);
	status = cublasSetMatrix(N, N, size, h_B, N, d_B, N);
	status = cublasSetMatrix(N, N, size, h_C, N, d_C, N);

	status  = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &al, d_A, N, d_B, N, &bet, d_C, N);

	status = cublasGetMatrix(N, N, size, d_C, N, h_C, N);

	checkResult(h_A, h_B, h_C, N);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	cublasDestroy(handle);

	free(h_A);
	free(h_B);
	free(h_C);

	return 0;
}
