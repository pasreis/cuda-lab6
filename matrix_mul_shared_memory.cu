/**
 * Inaki Urruta Sanchez
 * Pedro Alexandre Simoes dos Reis
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

#define BLOCK_SIZE 16

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
		for (int j = 0; j < dim ; j++) {
			M[i * dim + j] = (rand() % 10);
		}
	}
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

/**
 * Multiplies matrix left by the matrix right, both with dimensions dim and stores the result in matrix res
 * Operation is done in GPU
 */
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

/**
 * Multiplies matrix A by matrix B, both with dimension dim X dim and stores the result in matrix C with dimension dim X dim
 * Operation is done in CPU
 */
__host__
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

/**
 * Given two matrices A and B, both with dimensions dim X dim, prints in stdout if the result stored in matrix C with dimension dim X dim
 *   is the same as the result given in matrix C_cpu
 */
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
	// Set random seed
	srand(time(0));

	cudaError_t error;

	int deviceID;
	int numberOfSMs;

	error = cudaGetDevice(&deviceID);

	if (error != cudaSuccess) {
		printf("ERROR: %s\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	error = cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceID);

	if (error != cudaSuccess) {
		printf("ERROR: %s\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	// Matrix size definition and calculation
	const int N = 100;
	size_t size = N * N * sizeof(float);

	// Matrix allocation on Host
	float *h_A, *h_B, *h_C, *h_C_cpu;
	error = cudaMallocHost((void**) &h_A, size);

	if (error != cudaSuccess) {
		printf("ERROR: %s\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	error = cudaMallocHost((void**) &h_B, size);

	if (error != cudaSuccess) {
		printf("ERROR: %s\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	error = cudaMallocHost((void**) &h_C, size);

	if (error != cudaSuccess) {
		printf("ERROR: %s\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	error = cudaMallocHost((void**) &h_C_cpu, size);

	if (error != cudaSuccess) {
		printf("ERROR: %s\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	// Matrix initialization
	init(h_A, N);
	init(h_B, N);

	// Matrix allocation on Device
	float *d_A, *d_B, *d_C;
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

	// Copy matrixes A and B to device
	error = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

	if (error != cudaSuccess) {
		printf("ERROR: %s\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	error = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

	if (error != cudaSuccess) {
		printf("ERROR: %s\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	// Cuda layout definition
	dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3  blocksPerGrid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

	// Start timer
	double start = cpuTimer();

	matrixMul<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
	cudaDeviceSynchronize();

	// Stop timer
	double stop = cpuTimer();

	error = cudaGetLastError();

	if (error != cudaSuccess) {
		printf("ERROR: %s\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	// Print time interval
	double gpu_time = stop - start;
	printf("Matrix Multiplication @ GPU: %f ms \n", gpu_time);

	error = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

	if (error != cudaSuccess) {
		printf("ERROR: %s\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	// Start timer
	double begin = cpuTimer();

	matrixMulCPU(h_A, h_B, h_C_cpu, N);

	// Stop Timer
	double end = cpuTimer();

	// Print time interval
	double cpu_time = end - begin;
	printf("Matrix Multiplication @ CPU: %f ms \n", cpu_time);

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
