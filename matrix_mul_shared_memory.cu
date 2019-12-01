#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define BLOCK_SIZE 16

void initWith(float* M, int dim, float n) {
	for (int i = 0; i < dim; i++) {
		for (int j = 0; j < dim; j++) {
			M[i * dim + j] = n;
		}
	}
}

void init(float* M, int dim) {
	for (int i = 0; i < dim; i++) {
		for (int j = 0; j < dim ; j++) {
			M[i * dim + j] = rand();
		}
	}
}

__global__
void matrixMul(float* left, float* right, float* res, int dim) {
	int i, j, idx;
	float temp = 0;

	__shared__ float Left_shared_t [BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float Right_shared_t[BLOCK_SIZE][BLOCK_SIZE];

	// Row i of matrix left
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;


	for (int tileNUM = 0; tileNUM < gridDim.x; tileNUM++) {

		// Column j of matrix left
		j = tileNUM * BLOCK_SIZE + threadIdx.x;
		i = tileNUM * BLOCK_SIZE + threadIdx.y;
		// Load left[i][j] to shared mem

		idx = row * dim  + tileNUM * BLOCK_SIZE + threadIdx.x;

		if (idx >= dim * dim) {
			Left_shared_t[threadIdx.y][threadIdx.x] = 0;// Coalesced access
		} else {
			Left_shared_t[threadIdx.y][threadIdx.x] = left[row * dim + j];// Coalesced access
		}
		// Load right[i][j] to shared mem

		idx = (tileNUM * BLOCK_SIZE + threadIdx.y) * dim + col;

		if (idx >= dim * dim) {
			Right_shared_t[threadIdx.y][threadIdx.x] = 0;
		} else {
			Right_shared_t[threadIdx.y][threadIdx.x] = right[i * dim + col]; // Coalesced access
		}
		// Synchronize before computation
		__syncthreads();

		// Accumulate one tile of res from tiles of left and right in shared mem
		for (int k = 0; k < BLOCK_SIZE; k++) {

			temp += Left_shared_t[threadIdx.y][k] * Right_shared_t[k][threadIdx.x]; //no shared memory bank conflict
		}
		// Synchronize
		__syncthreads();
	}

	if ((row < dim) && (col < dim)) {
		// Store accumulated value to res
		res[row * dim + col] = temp;
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
	initWith(h_A, N, 1.0f);
	initWith(h_B, N, 1.0f);

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
	dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3  blocksPerGrid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

	matrixMul<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
	cudaDeviceSynchronize();

	cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
	cudaThreadSynchronize();

	initWith(h_C_cpu, N, 100.0f);
	checkResult(h_A, h_B, h_C, h_C_cpu, N);

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
