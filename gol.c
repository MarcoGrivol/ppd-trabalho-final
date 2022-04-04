// compile: gcc gol.c -o gol

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdbool.h>
#include <sys/time.h>

bool cellState(bool *grid, int row, int col, int N) {
    int sum =  
        grid[(row - 1) * N + col - 1] + // [-1, -1]
        grid[(row - 1) * N + col    ] + // [-1, 0]
        grid[(row - 1) * N + col + 1] + // [-1, 1]
        grid[ row      * N + col - 1] + // [0, -1]
        grid[ row      * N + col + 1] + // [0, 1]
        grid[(row + 1) * N + col - 1] + // [1, -1]
        grid[(row + 1) * N + col    ] + // [1, 0]
        grid[(row + 1) * N + col + 1];  // [1, 1]
    if (grid[row * N + col]) {
        // is alive
        if (sum == 2 || sum == 3) 
            return true;
        return false;
    }
    // is dead
    if (sum == 3)
        return true;
    return false;
}

bool cellStateBorder(bool *grid, int row, int col, int N) {
    int r, c, sum = 0;
    for (int i = row - 1; i <= row + 1; i++) {
        r = i < 0 ? N - 1 : i;
        r = i >= N ? 0 : r;
        for (int j = col - 1; j <= col + 1; j++) {
            c = j < 0 ? N - 1 : j;
            c = j >= N ? 0 : c;
            if (r != row || c != col) {
                sum += grid[r * N + c];
            }
        }
    }
    if (grid[row * N + col]) {
        // is alive
        if (sum == 2 || sum == 3)
            return true;
        return false;
    }
    // is dead
    if (sum == 3)
        return true;
    return false;
}

void updateState(bool *grid, bool *next_grid, int N) {
    bool state;
    for (int i = 1; i < N - 1; i++) {
        for (int j = 1; j < N - 1; j++) {
            next_grid[i * N + j] = cellState(grid, i, j, N);
        }
    }
    for (int j = 0; j < N; j++) {
        next_grid[0 * N + j] = cellStateBorder(grid, 0, j, N);
        next_grid[(N - 1) * N + j] = cellStateBorder(grid, N - 1, j, N);
    }
    for (int i = 1; i < N - 1; i++) {
        next_grid[i * N] = cellStateBorder(grid, i, 0, N);
        next_grid[i * N + (N - 1)] = cellStateBorder(grid, i, N - 1, N);
    }
    memcpy(grid, next_grid, sizeof(bool) * N * N);
}

int main() {
    srand(42);
    int N, G;
    // printf("Digite o tamanho do tabuleiro/grid (NxN): N=");
    // scanf("%d", &N);
    // printf("Digite a quantidade de gerações (1000 recomendado): G=");
    // scanf("%d", &G);
    N = 1000;
    G = 1000;
    int gens = G;

    bool *grid = malloc(N * N * sizeof(bool));
    bool *next_grid = malloc(N * N * sizeof(bool));
    for (int i = 0; i < N * N; i++) {
        grid[i] = (int) rand() % 2;
    }
    struct timeval t0, t1;
    float elapsed_time;
    gettimeofday(&t0, NULL);
    while (gens-- != 0) {
        updateState(grid, next_grid, N);
    }
    gettimeofday(&t1, NULL);
    elapsed_time = (t1.tv_sec - t0.tv_sec) + 
        (t1.tv_usec - t0.tv_usec) / 1000000.;
    printf("Time for grid(%d, %d) & %d generation: %.3f s\n", N, N, G, elapsed_time);
    printf("Generations per second: %.3f g/s\n", (float) G / elapsed_time);
    free(grid);
    free(next_grid);
    return 0;
}