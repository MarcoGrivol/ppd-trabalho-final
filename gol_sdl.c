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

void swap(bool **grid, bool **next_grid) {
    // Update grid to receive last iteration
    bool *aux = *grid;
    *grid = *next_grid;
    *next_grid = aux;
}

void updateState(int N, int G, bool *grid, bool *next_grid) {
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

int main(int argc, char **argv) {
    srand(time(NULL));

    int N, G;
    printf("Digite o tamanho do tabuleiro/grid (NxN): N=");
    scanf("%d", &N);
    if (N < 100) {
        printf("ERROR: N < 100 não gera uma boa visualização.\n");
        return -1;
    }
    printf("Digite a quantidade de gerações (1000 recomendado): G=");
    scanf("%d", &G);

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
    bool isRunning = true;
    while (isRunning) {
        while (SDL_PollEvent(&event) != 0) {
            if (event.type == SDL_QUIT) {
                isRunning = false;
            }
        }
        updateState(N, G, grid, next_grid);
        swap(&grid, &next_grid);
        copyGridToPixels(pixels, grid, N);
        SDL_Delay(10);
        SDL_UpdateWindowSurface(window);
        printf("\rGeração: %10d", G);
        fflush(stdout);
        if (--G == 0) {
        	isRunning = false;
        }
    }
    printf("\n");
    free(grid);
    free(next_grid);
    return 0;
}