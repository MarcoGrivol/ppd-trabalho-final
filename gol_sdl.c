// install SDL2: sudo apt-get install libsdl2-dev
// compile: gcc gol_sdl.c -o gol_sdl -lSDL2

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <SDL2/SDL.h>

void copyGridToPixels(int *pixels, bool *grid, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (grid[i * N + j]) 
                pixels[i * N + j] = 0xFFFFFF;
            else
                pixels[i * N + j] = 0x000000; 
        }
    }
}

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

void copyToGrid(bool *grid, bool *next_grid, int N) {
    for (int i = 0; i < N * N; i++) {
        grid[i] = next_grid[i];
    }
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
    copyToGrid(grid, next_grid, N);
}

int main() {
    srand(time(NULL));

    int N;
    printf("Digite o tamanho do tabuleiro/grid (NxN): N=");
    scanf("%d", &N);
    if (N < 500) {
        printf("ERROR: N < 500 não gera uma boa visualização.\n");
        return -1;
    }

    SDL_Init(SDL_INIT_VIDEO);
    SDL_Window *window = SDL_CreateWindow(
        "Conway's Game of Life", 
        SDL_WINDOWPOS_CENTERED,
        SDL_WINDOWPOS_CENTERED, 
        N, N, 
        SDL_WINDOW_SHOWN
    );
    // SDL_Window *renderer = SDL_CreateRenderer(window, -1, 0);
    SDL_Surface *surface = SDL_GetWindowSurface(window);
    int *pixels = (int *) surface->pixels;
    SDL_Event event;

    bool *grid = malloc(N * N * sizeof(bool));
    bool *next_grid = malloc(N * N * sizeof(bool));
    for (int i = 0; i < N * N; i++) {
        grid[i] = (int) rand() % 2;
    }
    bool isRunning = true;
    while (isRunning) {
        while (SDL_PollEvent(&event) != 0) {
            if (event.type == SDL_QUIT) {
                isRunning = false;
            }
        }
        updateState(grid, next_grid, N);
        copyGridToPixels(pixels, grid, N);
        SDL_Delay(10);
        SDL_UpdateWindowSurface(window);
    }
    free(grid);
    free(next_grid);
    return 0;
}