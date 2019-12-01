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
	for (int i = 0; i < dim; i++	) {
		for (int j = 0; j < dim; j++) {
			M[i * dim + j] = (rand() % 10);
		}
	}
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

	cudaDeviceProp prop;
	int numDevices = 0;

	error = cudaGetDeviceCount(&numDevices);

	if (error != cudaSuccess) {
		printf("ERROR: %s\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	int totalMemory = 0;

	for (int i = 0; i < numDevices; i++) {
		error = cudaGetDeviceProperties(&prop, i);

		if (error != cudaSuccess) {
			printf("ERROR: %s\n", cudaGetErrorString(error));
			exit(EXIT_FAILURE);
		}

		totalMemory += prop.totalGlobalMem;
	}

	// Matrix size definition and calculation
	const int N = 10;
	size_t size = N * N * sizeof(float);

	int allMatrixSizes = (N * N) * 3;
	if (allMatrixSizes > totalMemory) {
		printf("ERROR");
		exit(EXIT_FAILURE);
	}

	// Matrix allocation
	float *A, *B, *C, *C_cpu;
	error = cudaMallocManaged(&A, size);

	if (error != cudaSuccess) {
		printf("ERROR: %s\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	error = cudaMallocManaged(&B, size);

	if (error != cudaSuccess) {
		printf("ERROR: %s\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	error = cudaMallocManaged(&C, size);

	if (error != cudaSuccess) {
		printf("ERROR: %s\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	error =cudaMallocManaged(&C_cpu, size);

	if (error != cudaSuccess) {
		printf("ERROR: %s\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	// Matrix initialization
	init(A, N);
	init(B, N);

	// Cuda layout definition
	dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 blocksPerGrid((N + BLOCK_SIZE - 1) /  BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

	// Start timer
	double start = cpuTimer();
	matrixMul<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
	cudaDeviceSynchronize();
	// Stop timer
	double stop = cpuTimer();

	error = cudaGetLastError();

	if (error != cudaSuccess) {
		printf("ERROR: %s\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	// Print time interval
	float gpu_milliseconds = stop - start;

	printf("Matrix Multiplication @ GPU: %f ms\n", gpu_milliseconds);

	// Start timer
	double begin = cpuTimer();

	// Matrix multiplication in CPU
	matrixMulCPU(A, B, C_cpu, N);

	// Stop timer
	double end = cpuTimer();

	// Print time interval
	float cpu_milliseconds = end - begin;
	printf("Matrix Multiplication @ CPU: %f ms\n", cpu_milliseconds);

	checkResult(A, B, C, C_cpu, N);

	// Free memory
	cudaFree(A);
	cudaFree(B);
	cudaFree(C);
	cudaFree(C_cpu);

	return 0;
}
