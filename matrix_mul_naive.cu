#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define BLOCK_SIZE 16

void initWith(float* M, int dim) {
	for (int i = 0; i < dim; i++) {
		for (int j = 0; j < dim; j++) {
			M[i * dim + j] = rand();
		}
	}
}

__global__
void matrixMul(float* A, float* B, float* res, int dim) {
	__shared__ float tile_A[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float tile_B[BLOCK_SIZE][BLOCK_SIZE];

	int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
	int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

	float tmp = 0.0;
	int idx;

	for (int i = 0; i < gridDim.x; i++) {
		idx = row * dim + i * BLOCK_SIZE + threadIdx.x;

		if (idx >= dim * dim) {
			tile_A[threadIdx.y][threadIdx.x] = 0;
		} else {
			tile_A[threadIdx.y][threadIdx.x] = A[idx];
		}

		idx = (i * BLOCK_SIZE + threadIdx.y) * dim + col;

		if (idx >= dim * dim) {
			tile_B[threadIdx.y][threadIdx.x] = 0;
		} else {
			tile_B[threadIdx.y][threadIdx.x] = B[idx];
		}

		__syncthreads();

		for (int j = 0; j < BLOCK_SIZE; j++) {
			tmp += tile_A[threadIdx.y][j] * tile_B[j][threadIdx.x];
		}

		__syncthreads();
	}

	if (row < dim && col < dim) {
		res[row * dim + col] = tmp;
	}
}

void matrixMulCPU(float* A, float* B, float* C, int dim) {
	for (int i = 0; i < dim; i++) {
		for (int j = 0; j < dim; j++) {
			float tmp = 0.0;
			for (int k = 0; k < dim; k++) {
				tmp += A[i * dim + k] * B[k * dim + j];
			}
			C[i * dim + j]  = tmp;
		}
	}
}

void checkResult(float* A, float* B, float* C, float* C_cpu, int dim) {
	for (int i = 0; i < dim; i++) {
		for (int j = 0; j < dim; j++) {
			if (C_cpu[i * dim + j] != C[i * dim + j]) {
				printf("ERROR: Incorrect Results!\n");
				return;
			}
		}
	}

	printf("Everything is OK! :D\n");
}

int main(int argc, char** argv) {
	srand(time(0));

	cudaDeviceProp prop;
	int numDevices = 0;

	cudaGetDeviceCount(&numDevices);

	int totalMemory = 0;

	for (int i = 0; i < numDevices; i++) {
		cudaGetDeviceProperties(&prop, i);
		totalMemory += prop.totalGlobalMem;
	}

	// Matrix size definition and calculation
	const int N = 100;
	size_t size = N * N * sizeof(float);

	int allMatrixSizes = (N * N) * 3;
	if (allMatrixSizes > totalMemory) {
		printf("ERROR");
		exit(EXIT_FAILURE);
	}

	// Matrix allocation
	float *A, *B, *C, *C_cpu;
	cudaMallocManaged(&A, size);
	cudaMallocManaged(&B, size);
	cudaMallocManaged(&C, size);
	cudaMallocManaged(&C_cpu, size);

	// Matrix initialization
	initWith(A, N);
	initWith(B, N);

	// Cuda layout definition
	unsigned int grid = (N + N - 1) / N;
	dim3 dimGrid(grid, grid);
	dim3 dimBlock(N, N);

	matrixMulCPU(A, B, C_cpu, N);
	matrixMul<<<dimGrid, dimBlock>>>(A, B, C, N);
	cudaDeviceSynchronize();

	checkResult(A, B, C, C_cpu, N);

	// Free memory
	cudaFree(A);
	cudaFree(B);
	cudaFree(C);
	cudaFree(C_cpu);

	return 0;
}
