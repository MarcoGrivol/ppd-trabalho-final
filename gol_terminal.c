// compile: gcc gol_terminal.c -o gol_terminal

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __linux__
    #define console NULL
    void mysleep(msec) { usleep(msec * 1000); };
    void clrscr() { printf("\e[1;1H\e[2J"); }
#elif _WIN32
    #include <windows.h>
    void mysleep(int msec) { Sleep(msec); }
    void clrscr(HANDLE hConsole) {
        // source: https://docs.microsoft.com/en-us/windows/console/clearing-the-screen
        COORD coordScreen = { 0, 0 };    // home for the cursor
        DWORD cCharsWritten;
        CONSOLE_SCREEN_BUFFER_INFO csbi;
        DWORD dwConSize;

        // Get the number of character cells in the current buffer.
        if (!GetConsoleScreenBufferInfo(hConsole, &csbi))
        {
            return;
        }

        dwConSize = csbi.dwSize.X * csbi.dwSize.Y;

        // Fill the entire screen with blanks.
        if (!FillConsoleOutputCharacter(hConsole,        // Handle to console screen buffer
                                        (TCHAR)' ',      // Character to write to the buffer
                                        dwConSize,       // Number of cells to write
                                        coordScreen,     // Coordinates of first cell
                                        &cCharsWritten)) // Receive number of characters written
        {
            return;
        }

        // Get the current text attribute.
        if (!GetConsoleScreenBufferInfo(hConsole, &csbi))
        {
            return;
        }

        // Set the buffer's attributes accordingly.
        if (!FillConsoleOutputAttribute(hConsole,         // Handle to console screen buffer
                                        csbi.wAttributes, // Character attributes to use
                                        dwConSize,        // Number of cells to set attribute
                                        coordScreen,      // Coordinates of first cell
                                        &cCharsWritten))  // Receive number of characters written
        {
            return;
        }
        // Put the cursor at its home coordinates.
        SetConsoleCursorPosition(hConsole, coordScreen);
    }
#endif


void printGrid(bool *grid, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (grid[i * N + j]) printf("O ");
            else printf(". ");
            // printf("%3d", grid[i * N + j]);
        }
        printf("\n");
    }
    // printf("\n%3d", grid[(N - 1) * N + N - 1]);
    // for (int j = 0; j < N; j++) {
    //     printf("%3d", grid[(N - 1) * N + j]);
    // }
    // printf("%3d\n", grid[(N - 1) * N]);
    // for (int i = 1; i < N; i++) {
    //     printf("%3d", grid[i * N + N - 1]);
    //     printf("%18d\n", grid[i * N]);
    // }
    // printf("%3d", grid[N - 1]);
    // for (int j = 0; j < N; j++) {
    //     printf("%3d", grid[j]);
    // }
    // printf("%3d\n", grid[0]);
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
    #ifdef _WIN32
        HANDLE console;
        console = GetStdHandle(STD_OUTPUT_HANDLE);
    #endif
    srand(time(NULL));
    int N;
    printf("Digite o tamanho do tabuleiro/grid (NxN): N=");
    scanf("%d", &N);
    clrscr(console);

    bool *grid = malloc(N * N * sizeof(bool));
    bool *next_grid = malloc(N * N * sizeof(bool));
    for (int i = 0; i < N * N; i++) {
        grid[i] = (int) rand() % 2;
        grid[i] = false;
    }
    grid[0 * N + 1] = true;
    grid[1 * N + 2] = true;
    grid[2 * N] = true;
    grid[2 * N + 1] = true;
    grid[2 * N + 2] = true;
    printGrid(grid, N);
    // mysleep(3000);
    clrscr(console);
    while (true) {
        updateState(grid, next_grid, N);
        printGrid(grid, N);
        mysleep(250);
        clrscr(console);
    }
    free(grid);
    free(next_grid);
    return 0;
}