#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <sys/time.h>

__global__ void updateState(bool *d_grid, bool *d_next_grid, int N) {
    // gets the row and col from thread and block index
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 0 && i < N && j >= 0 && j < N) {
        int sum =  
            d_grid[(i - 1 + N) % N * N + (j - 1 + N) % N] + // [-1, -1]
            d_grid[(i - 1 + N) % N * N +  j             ] + // [-1,  0]
            d_grid[(i - 1 + N) % N * N + (j + 1 + N) % N] + // [-1,  1]
            d_grid[ i              * N + (j - 1 + N) % N] + // [ 0, -1]
            d_grid[ i              * N + (j + 1 + N) % N] + // [ 0,  1]
            d_grid[(i + 1 + N) % N * N + (j - 1 + N) % N] + // [ 1, -1]
            d_grid[(i + 1 + N) % N * N +  j             ] + // [ 1,  0]
            d_grid[(i + 1 + N) % N * N + (j + 1 + N) % N];  // [ 1,  1]
        if (d_grid[i * N + j]) {
            // is alive
            if (sum == 2 || sum == 3) d_next_grid[i * N + j] = true;
            else d_next_grid[i * N + j] = false;
        } 
        else {
            // is dead
            if (sum == 3) d_next_grid[i * N + j] = true;
            else d_next_grid[i * N + j] = false;
        }
    }
    return;
}

void swap(bool **d_grid, bool **d_next_grid) {
    // Update grid to receive last iteration
    bool *aux = *d_grid;
    *d_grid = *d_next_grid;
    *d_next_grid = aux;
}

void runAllGenerations(int N, int G, bool *d_grid, bool *d_next_grid) {

    // defines the # of threads required
    int threads = 32; // 32 * 32 = 1024 -> max # of threads per block
    dim3 blocks (ceil((float) N / threads), ceil((float) N / threads)); // blocks in grid
    dim3 threads_per_block (threads, threads);
    printf("Threads: %d\n", 
        threads_per_block.x * threads_per_block.y * blocks.x * blocks.y
    );
    printf("blocks (%d, %d)\nthreads_per_block (%d, %d)\n", 
        blocks.x, blocks.y,
        threads_per_block.x, threads_per_block.y
    );

    // Updates the state of the grid G times
    for (int i = 0; i < G; i++) {
        updateState<<<blocks, threads_per_block>>>(d_grid, d_next_grid, N);
        cudaDeviceSynchronize();
        swap(&d_grid, &d_next_grid);
    }
}

int main(int argc, char *argv[]) {
    srand(42);
    
    if (argc != 3) {
        printf("gol_sequential.c requires N and G integers.\n");
        return -1;
    }
    int N = atoi(argv[1]); // size of grid (N x N)
    int G = atoi(argv[2]);     // # of generations

    bool *grid = (bool *) malloc(N * N * sizeof(bool));
    bool *next_grid = (bool *) malloc(N * N * sizeof(bool));
    // randomly initialize grid
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            grid[i * N + j] = (int) rand() % 2;
        }
    }

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    // start timer
    cudaEventRecord(t0);

    // device memory allocation
    bool *d_grid, *d_next_grid;
    cudaMalloc(&d_grid, N * N * sizeof(bool));
    cudaMalloc(&d_next_grid, N * N * sizeof(bool));
    // copy grid to device
    cudaMemcpy(d_grid, grid, N * N * sizeof(bool), cudaMemcpyHostToDevice);

    runAllGenerations(N, G, d_grid, d_next_grid);

    // copy grid from device to host
    cudaMemcpy(grid, d_next_grid, N * N * sizeof(bool), cudaMemcpyDeviceToHost);
    cudaFree(d_grid);
    cudaFree(d_next_grid);

    cudaEventRecord(t1);
    // end timer

    cudaEventSynchronize(t1);
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, t0, t1);
    elapsed_time /= 1000.;
    printf("Time for grid %d x %d and %d generations: %.3f s\n", N, N, G, elapsed_time);
    printf("Generations per second: %.1f g/s\n", (float) G / elapsed_time);
    //verify integrity
    bool *correct_grid = (bool *) malloc(N * N * sizeof(bool));
    FILE *fp;
    fp = fopen("correct_grid.bin", "rb");
    int rc = fread(correct_grid, sizeof(*correct_grid), N * N, fp);
    fclose(fp);
    if (rc == N * N) {
        int errors = 0;
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                if (correct_grid[i * N + j] != grid[i * N + j]) {
                    errors++;
                }
            }
        }
        printf("%d errors\n", errors);
    }
    else {
        printf("ERROR in reading file\n");
        printf("Successfully read %d elements\n", rc);
    }
    
    free(grid);
    free(next_grid);
    return 0;
}