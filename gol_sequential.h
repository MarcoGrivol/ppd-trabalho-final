#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

void startProcedures() {};
void endProcedures() {};

void updateState(bool *grid, bool *next_grid, int N);
bool cellState(bool *grid, int row, int col, int N);
bool cellStateBorder(bool *grid, int row, int col, int N);

void runAllGenerations(int N, int G, bool *grid, bool *next_grid) {
    for (int i = 0; i < G; i++) {
        updateState(grid, next_grid, N);
    }
}

bool cellState(bool *grid, int row, int col, int N) {
    int sum =  
        grid[(row - 1) * N + col - 1] + // [-1, -1]
        grid[(row - 1) * N + col    ] + // [-1,  0]
        grid[(row - 1) * N + col + 1] + // [-1,  1]
        grid[ row      * N + col - 1] + // [ 0, -1]
        grid[ row      * N + col + 1] + // [ 0,  1]
        grid[(row + 1) * N + col - 1] + // [ 1, -1]
        grid[(row + 1) * N + col    ] + // [ 1,  0]
        grid[(row + 1) * N + col + 1];  // [ 1,  1]
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
    // copy borders
    for (int i = 0; i < N; i++) {
        next_grid[i] = next_grid[(N - 2) * N + i];
        next_grid[(N - 1) * N + i] = next_grid[N + i];
    }
    for (int j = 0; j < N; j++) {
        next_grid[j * N] = next_grid[j * N + N - 2];
        next_grid[j * N + N - 1] = next_grid[j * N + 1];
    }
    memcpy(grid, next_grid, sizeof(bool) * N * N);
}
