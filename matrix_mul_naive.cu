#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

#define BLOCK_SIZE 16

void initWith(float* M, int dim, float n) {
	for (int i = 0; i < dim; i++) {
		for (int j = 0; j < dim; j++) {
			M[i * dim + j] = n;
		}
	}
}

void init(float* M, int dim) {
	for (int i = 0; i < dim; i++	) {
		for (int j = 0; j < dim; j++) {
			M[i * dim + j] = (rand() % 10);
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
	/*__shared__ float tile_A[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float tile_B[BLOCK_SIZE][BLOCK_SIZE];

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	float tmp = 0.0;
	int idx;

	int row_tile, col_tile;

	for (int i = 0; i < gridDim.x; i++) {*/


		/*col_tile = i * BLOCK_SIZE + threadIdx.x;
		row_tile = i * BLOCK_SIZE + threadIdx.y;

		tile_A[threadIdx.y][threadIdx.x] = A[row_tile * dim + col];
		tile_B[threadIdx.y][threadIdx.x] = B[row * dim * col_tile];

		__syncthreads();*/
		/*
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
		}*/

		/*__syncthreads();

		for (int j = 0; j < BLOCK_SIZE; j++) {
			tmp += tile_A[j][threadIdx.x] * tile_B[threadIdx.y][j];
		}

		__syncthreads();*/
	}
/*
	if (row < dim && col < dim) {
		res[row * dim + col] = tmp;
	}
	__syncthreads();*/
	//res[row * dim + col] = tmp;

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
	init(A, N);
	init(B, N);

	// Cuda layout definition
	dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 blocksPerGrid((N + BLOCK_SIZE - 1) /  BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);


	// Start timer
	double start = cpuTimer();
	//matrixMulCPU(A, B, C_cpu, N);
	matrixMul<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
	cudaDeviceSynchronize();
	// Stop timer
	double stop = cpuTimer();

	// Print time interval
	float gpu_milliseconds = stop - start;

	printf("Matrix Multiplication @ GPU: %f ms\n", gpu_milliseconds);

	// Start timer
	double begin = cpuTimer();

	// Matrix multiplication in CPU
	matrixMulCPU(A, B, C_cpu, N);

	// Stop timer
	double end = cpuTimer();

	// Print time inerval
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
