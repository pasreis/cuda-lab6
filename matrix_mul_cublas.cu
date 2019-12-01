#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
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
			M[i * dim + j] = (rand() % 10);
		}
	}
}

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

double cpuTimer() {
	struct timeval clock;
	gettimeofday(&clock, NULL);
	return ((double) clock.tv_sec + (double) clock.tv_usec * 1e-6);
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

	float* d_A, *d_B, *d_C;

	error = cudaMalloc((void**) &d_A, size);
	error = cudaMalloc((void**) &d_B, size);
	error = cudaMalloc((void**) &d_C, size);

	status = cublasCreate(&handle);

	float al = 1.0f, bet = 0.0f;

	status = cublasSetMatrix(N, N, sizeof(float), h_A, N, d_A, N);
	status = cublasSetMatrix(N, N, sizeof(float), h_B, N, d_B, N);
	status = cublasSetMatrix(N, N, sizeof(float), h_C, N, d_C, N);

	// Star timer
	double start = cpuTimer();
	status  = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &al, d_B, N, d_A, N, &bet, d_C, N);
	// Stop timer
	double stop = cpuTimer();

	// Print time interval
	double gpu_milliseconds = stop - start;
	printf("Matrix Multiplication @ GPU: %f ms\n", gpu_milliseconds);

	cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

	checkResult(h_A, h_B, h_C, h_C_cpu, N);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	cublasDestroy(handle);

	free(h_A);
	free(h_B);
	free(h_C);

	return 0;
}
