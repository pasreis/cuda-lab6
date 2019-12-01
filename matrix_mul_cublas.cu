/**
 * Inaki Urruta Sanchez
 * Pedro Alexandre Simoes dos Reis
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

/**
 * Initialize matrix M with dimension dim with n in all matrix's entries
 */
void initWith(float* M, int dim, float n) {
	for (int i = 0; i < dim; i++) {
		for (int j = 0; j < dim; j++) {
			M[i * dim + j] = n;
		}
	}
}

/**
 * Initialize matrix M with dimension dim with a random number between 0 and 9 in all matrix's entries
 */
void init(float* M, int dim) {
	for (int i = 0; i < dim; i++) {
		for (int j = 0; j < dim; j++) {
			M[i * dim + j] = (rand() % 10);
		}
	}
}

/**
 * Multiplies matrix A by matrix B, both with dimension dim X dim and stores the result in matrix C with dimension dim X dim
 * Operation is done in CPU
 */
__host__
void matrixMulCPU(float* A, float* B, float* C_cpu, int dim) {
	for (int i = 0; i < dim; i++) {
		for (int j = 0; j < dim; j++) {
			float tmp = 0.0f;
			for (int k = 0; k < dim; k++) {
				tmp += A[i * dim + k] * B[k * dim + j];
			}
			C_cpu[i * dim + j] = tmp;
		}
	}
}

/**
 * Given two matrices A and B, both with dimensions dim X dim, prints in stdout if the result stored in matrix C with dimension dim X dim
 *   is the same as the result given in matrix C_cpu
 */
void checkResult(float* A, float* B, float* C, float* C_cpu, int dim) {
	for (int i = 0; i < dim; i++) {
		for (int j = 0; j < dim; j++) {
			if (abs(C[i * dim + j] - C_cpu[i * dim + j]) > 0.001) {
				printf("ERROR: Incorrect Results!\n");
				return;
			}
		}
	}

	printf("Everything is OK! :D\n");
}

/**
 * Returns the current time in milliseconds
 * Used to calculate elapsed time
 */
double cpuTimer() {
	struct timeval clock;
	gettimeofday(&clock, NULL);
	return ((double) clock.tv_sec + (double) clock.tv_usec * 1e-6);
}

int main(int argc, char** argv) {
	// Set random seed
	srand(time(0));

	cudaError_t error;
	cublasStatus_t status;
	cublasHandle_t handle;

	int N = 100;
	size_t size = N * N * sizeof(float);

	// Matrix declaration and allocation on host
	float* h_A, *h_B, *h_C, *h_C_cpu;
	h_A = (float*) malloc(size);
	h_B = (float*) malloc(size);
	h_C = (float*) malloc(size);
	h_C_cpu = (float*) malloc(size);

	init(h_A, N);
	init(h_B, N);

	// Start timer
	double begin = cpuTimer();

	matrixMulCPU(h_A, h_B, h_C_cpu, N);

	// Stop timer
	double end = cpuTimer();

	// Print time interval
	double cpu_milliseconds = end - begin;
	printf("Matrix Multiplication @ CPU: %f\n", cpu_milliseconds);

	// Matrix declaration and allocation on device
	float* d_A, *d_B, *d_C;
	error = cudaMalloc((void**) &d_A, size);

	if (error != cudaSuccess) {
		printf("ERROR: %s\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	error = cudaMalloc((void**) &d_B, size);

	if (error != cudaSuccess) {
		printf("ERROR: %s\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	error = cudaMalloc((void**) &d_C, size);

	if (error != cudaSuccess) {
		printf("ERROR: %s\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	status = cublasCreate(&handle);

	if (status != CUBLAS_STATUS_SUCCESS) {
		printf("ERROR on Cublas\n");
		exit(EXIT_FAILURE);
	}

	float al = 1.0f, bet = 0.0f;

	status = cublasSetMatrix(N, N, sizeof(float), h_A, N, d_A, N);

	if (status != CUBLAS_STATUS_SUCCESS) {
		printf("ERROR on Cublas\n");
		exit(EXIT_FAILURE);
	}

	status = cublasSetMatrix(N, N, sizeof(float), h_B, N, d_B, N);

	if (status != CUBLAS_STATUS_SUCCESS) {
		printf("ERROR on Cublas\n");
		exit(EXIT_FAILURE);
	}

	status = cublasSetMatrix(N, N, sizeof(float), h_C, N, d_C, N);

	if (status != CUBLAS_STATUS_SUCCESS) {
		printf("ERROR on Cublas\n");
		exit(EXIT_FAILURE);
	}

	// Star timer
	double start = cpuTimer();
	status  = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &al, d_B, N, d_A, N, &bet, d_C, N);
	// Stop timer
	double stop = cpuTimer();

	if (status != CUBLAS_STATUS_SUCCESS) {
		printf("ERROR on Cublas\n");
		exit(EXIT_FAILURE);
	}

	// Print time interval
	double gpu_milliseconds = stop - start;
	printf("Matrix Multiplication @ GPU: %f ms\n", gpu_milliseconds);

	error = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

	if (error != cudaSuccess) {
		printf("ERROR: %s\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	checkResult(h_A, h_B, h_C, h_C_cpu, N);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	cublasDestroy(handle);

	free(h_A);
	free(h_B);
	free(h_C);

	return EXIT_SUCCESS;
}
