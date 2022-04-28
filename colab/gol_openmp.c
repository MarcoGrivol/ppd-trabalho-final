#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <sys/time.h>

void swap(bool **grid, bool **next_grid) {
    // Update grid to receive last iteration
    bool *aux = *grid;
    *grid = *next_grid;
    *next_grid = aux;
}

void runAllGenerations(int N, int G, bool *grid, bool *next_grid) {
    // Updates the state of the grid G times.
    #pragma omp parallel shared(grid, next_grid, N, G)
    {
        for (int g = 0; g < G; g++) {
            // for each iteration
            #pragma omp for schedule(static)
            for (int i = 1; i < N - 1; i++) {
                // for each row, excluding borders
                for (int j = 1; j < N - 1; j++) {
                    // for each col, excluding borders
                    // counts the number of live neighbors
                    int sum =  
                        grid[(i - 1) * N + j - 1] + // [-1, -1]
                        grid[(i - 1) * N + j    ] + // [-1,  0]
                        grid[(i - 1) * N + j + 1] + // [-1,  1]
                        grid[ i      * N + j - 1] + // [ 0, -1]
                        grid[ i      * N + j + 1] + // [ 0,  1]
                        grid[(i + 1) * N + j - 1] + // [ 1, -1]
                        grid[(i + 1) * N + j    ] + // [ 1,  0]
                        grid[(i + 1) * N + j + 1];  // [ 1,  1]
                    // updates cell state
                    if (grid[i * N + j]) {
                        // is alive
                        if (sum == 2 || sum == 3) next_grid[i * N + j] = true;
                        else next_grid[i * N + j] = false;
                    } 
                    else {
                        // is dead
                        if (sum == 3) next_grid[i * N + j] = true;
                        else next_grid[i * N + j] = false;
                    }
                }
            }
            // wrap-around padding
            // copy first and last row borders
            #pragma omp for schedule(static)
            for (int col = 1; col < N - 1; col++) {
                next_grid[col] = next_grid[(N - 2) * N + col];
                next_grid[(N - 1) * N + col] = next_grid[N + col];
            }
            // copy left and right borders
            #pragma omp for schedule(static)
            for (int row = 0; row < N; row++) {
                next_grid[row * N] = next_grid[row * N + N - 2];
                next_grid[row * N + N - 1] = next_grid[row * N + 1];
            }
            #pragma omp single
            {
                swap(&grid, &next_grid);
            }
        }
    }
}

int main(int argc, char *argv[]) {
    srand(42);
    
    if (argc != 3) {
        printf("gol_sequential.c requires N and G integers.\n");
        return -1;
    }
    int N = atoi(argv[1]) + 2; // size of grid (N x N)
    int G = atoi(argv[2]);

    bool *grid = malloc(N * N * sizeof(bool));
    bool *next_grid = malloc(N * N * sizeof(bool));
    // randomly initialize grid
    for (int i = 1; i < N - 1; i++) {
        for (int j = 1; j < N - 1; j++) {
            grid[i * N + j] = (int) rand() % 2;
        }
    }
    // wrap-around padding
    // copy first and last row borders
    for (int col = 0; col < N; col++) {
        grid[col] = grid[(N - 2) * N + col];
        grid[(N - 1) * N + col] = grid[N + col];
    }
    // copy left and right borders
    for (int row = 0; row < N; row++) {
        grid[row * N] = grid[row * N + N - 2];
        grid[row * N + N - 1] = grid[row * N + 1];
    }

    struct timeval t0, t1;
    // start timer
    gettimeofday(&t0, NULL);

    runAllGenerations(N, G, grid, next_grid);
    if (G % 2 != 0) swap(&grid, &next_grid);
    
    gettimeofday(&t1, NULL);
    // end timer

    float elapsed_time;
    elapsed_time = 
        (t1.tv_sec - t0.tv_sec)
        + (t1.tv_usec - t0.tv_usec) / 1000000.;
    printf("Time for grid %d x %d and %d generations: %.3f s\n", N-2, N-2, G, elapsed_time);
    printf("Generations per second: %.2f g/s\n", (float) G / elapsed_time);
    
    //verify integrity
    bool *correct_grid = malloc(N * N * sizeof(bool));
    FILE *fp;
    fp = fopen("correct_grid.bin", "rb");
    int rc = fread(correct_grid, sizeof(*grid), N * N, fp);
    fclose(fp);
    if (rc == N * N) {
        int errors = 0;
        for (int i = 0; i < N * N; i++) {
            if (correct_grid[i] != grid[i]) {
                errors++;
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
