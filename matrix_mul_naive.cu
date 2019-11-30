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
	srand(time(0));

	int deviceID;
	int numberOfSMs;

	cudaGetDevice(&deviceID);
	cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceID);

	// Matrix size definition and calculation
	const int N = 100;
	size_t size = N * N * sizeof(float);

	// Matrix allocation on Host
	float *h_A, *h_B, *h_C, *h_C_cpu;
	cudaMallocHost((void**) &h_A, size);
	cudaMallocHost((void**) &h_B, size);
	cudaMallocHost((void**) &h_C, size);
	cudaMallocHost((void**) &h_C_cpu, size);

	// Matrix initialization
	initWith(h_A, N);
	initWith(h_B, N);

	// Matrix allocation on Device
	float *d_A, *d_B, *d_C;
	cudaMalloc((void**) &d_A, size);
	cudaMalloc((void**) &d_B, size);
	cudaMalloc((void**) &d_C, size);

	// Copy matrixes A and B to device
	cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

	// Cuda layout definition
	unsigned int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
	dim3 dimGrid(grid, grid);
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

	matrixMul<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);
	cudaDeviceSynchronize();

	cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
	cudaThreadSynchronize();

	checkResult(h_A, h_B, h_C, N);

	// Free memory
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	cudaFreeHost(h_A);
	cudaFreeHost(h_B);
	cudaFreeHost(h_C);
	cudaFreeHost(h_C_cpu);

	return 0;
}
