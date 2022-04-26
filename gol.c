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

void updateState(int N, bool *grid, bool *next_grid) {
    // for each iteration
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
    for (int i = 1; i < N - 1; i++) {
        next_grid[i] = next_grid[(N - 2) * N + i];
        next_grid[(N - 1) * N + i] = next_grid[N + i];
    }
    // copy left and right borders
    for (int j = 0; j < N; j++) {
        next_grid[j * N] = next_grid[j * N + N - 2];
        next_grid[j * N + N - 1] = next_grid[j * N + 1];
    }
}

void printGrid(bool *grid, int N) {
    for (int i = 1; i < N - 1; i++) {
        for (int j = 1; j < N - 1; j++) {
            if (grid[i * N + j]) printf("O ");
            else printf(". ");
        }
        printf("\n");
    }
}

void runAllGenerations(int N, int G, bool *grid, bool *next_grid) {
    /* GLIDER
    */
    for (int i = 0; i < N * N; i++) {
        // grid[i] = (int) rand() % 2;
        grid[i] = false;
    }
    grid[1 * N + 2] = true;
    grid[2 * N + 3] = true;
    grid[3 * N + 1] = true;
    grid[3 * N + 2] = true;
    grid[3 * N + 3] = true;

    system("clear");
    printGrid(grid, N);
    usleep(100000);
    system("clear");

    for (int i = 0; i < G; i++) {
        updateState(N, grid, next_grid);
        swap(&grid, &next_grid);
        printGrid(grid, N);
        usleep(100000);
        system("clear");
    }
}

int main() {
    srand(42);
    
    int N = 10 + 2;
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
 
    runAllGenerations(N, G, grid, next_grid);

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