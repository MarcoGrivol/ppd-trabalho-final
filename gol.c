#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <sys/time.h>

int main() {
    srand(42);
    
    int N = 50 + 2;
    int G = 1000;

    bool *grid = malloc(N * N * sizeof(bool));
    bool *next_grid = malloc(N * N * sizeof(bool));
    for (int i = 1, c=0; i < N - 1; i++) {
        for (int j = 1; j < N - 1; j++) {
            grid[i * N + j] = (int) rand() % 2;
        }
    }
    for (int col = 0; col < N; col++) {
        grid[col] = grid[(N - 2) * N + col];
        grid[(N - 1) * N + col] = grid[N + col];
    }
    for (int row = 0; row < N; row++) {
        grid[row * N] = grid[row * N + N - 2];
        grid[row * N + N - 1] = grid[row * N + 1];
    }

    struct timeval t0, t1;
    // start timer
    gettimeofday(&t0, NULL);
 
    startProcedures();
    runAllGenerations(N, G, grid, next_grid);
    endProcedures();

    gettimeofday(&t1, NULL);
    // end timer

    float elapsed_time;
    elapsed_time = 
        (t1.tv_sec - t0.tv_sec)
        + (t1.tv_usec - t0.tv_usec) / 1000000.;
    printf("Time for grid %d x %d and %d generations: %.3f s\n", N, N, G, elapsed_time);
    printf("Generations per second: %.3f g/s\n", (float) G / elapsed_time);
    
    free(grid);
    free(next_grid);
    return 0;
}