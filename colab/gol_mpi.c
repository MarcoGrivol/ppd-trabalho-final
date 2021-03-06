#include <mpi.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <sys/time.h>

// 
struct info {
    int start;
    int end;
    int send_first_to;
    int send_last_to;
} t_info;

void swap(bool **grid, bool **next_grid) {
    // Update grid to receive last iteration
    bool *aux = *grid;
    *grid = *next_grid;
    *next_grid = aux;
}

void copyBorders(bool *grid, int N) {
    // wrap-around padding
    // copy first and last row borders
    for (int col = 1; col < N - 1; col++) {
        grid[col] = grid[(N - 2) * N + col];
        grid[(N - 1) * N + col] = grid[N + col];
    }
    // copy left and right borders
    for (int row = 0; row < N; row++) {
        grid[row * N] = grid[row * N + N - 2];
        grid[row * N + N - 1] = grid[row * N + 1];
    }
}

void sendGrid(int numtasks, struct info *t_infos, bool *grid, int N) {
    // divides and sends the grid for each task
    for (int i = 1; i < numtasks; i++) {
        int start = t_infos[i - 1].start;
        int end = t_infos[i - 1].end;
        int numrows = end - start + 1;
        for (int r = start; r <= end; r++) 
            MPI_Send(&grid[r * N], N, MPI_C_BOOL, i, 1, MPI_COMM_WORLD);
    }
}

void recvGrid(int numtasks, struct info *t_infos, bool *grid, int N) {
    // receives the updated grid portion from each task
    for (int i = 1; i < numtasks; i++) {
        int start = t_infos[i - 1].start;
        int end = t_infos[i - 1].end;
        int numrows = end - start + 1;
        for (int r = start + 1; r < end; r++)
            MPI_Recv(&grid[r * N], N, MPI_C_BOOL, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

void masterRoutine(int numtasks, int N, int G) {
    // responsible for dividing, sending and receiving the grid for each task

    bool *grid = (bool *) malloc(N * N * sizeof(bool));
    // ramdomly initializes the grid
    
    for (int i = 1; i < N - 1; i++) {
        for (int j = 1; j < N - 1; j++) {
            grid[i * N + j] = (int) rand() % 2;
        }
    }
    copyBorders(grid, N);

    struct timeval t0, t1;
    // start timer
    gettimeofday(&t0, NULL);

    // Task Division
    int numrows = N / (numtasks - 1);
    struct info t_infos[numtasks - 1];
    int start = 0;
    int end = numrows + 1;
    for (int i = 1; i < numtasks; i++) {
        t_infos[i - 1].start = start;
        t_infos[i - 1].end = end;
        t_infos[i - 1].send_first_to = i - 1 == 0 ? numtasks - 1 : i - 1;
        t_infos[i - 1].send_last_to = i + 1 == numtasks ? 1 : i + 1;
        MPI_Send(&t_infos[i - 1], sizeof(t_infos[i - 1]), MPI_INT, i, 1, MPI_COMM_WORLD);
        start = end - 1;
        if (i == numtasks - 2)
            end = N - 1;    
        else
            end += numrows;
    }

    // Update the grid
    sendGrid(numtasks, t_infos, grid, N);
    recvGrid(numtasks, t_infos, grid, N);
    copyBorders(grid, N);

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
    free(correct_grid);
}

void workerRoutine(int rank, int N, int G) {
    // responsible for checking cell state from the received grid portion

    // gets the information from master 
    MPI_Recv(&t_info, sizeof(t_info), MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    int numrows = t_info.end - t_info.start + 1;

    // allocate necessary memory
    bool *grid = (bool *) malloc(numrows * N * sizeof(bool));;
    bool *next_grid = (bool *) malloc(numrows * N * sizeof(bool));

    for (int i = 0; i < numrows; i++) 
        MPI_Recv(&grid[i * N], N, MPI_C_BOOL, 0, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    
    // iterate over all generations
    for (int g = 0; g < G; g++) {
        // Update State
        for (int i = 1; i < numrows - 1; i++) {
            for (int j = 1; j < N - 1; j++) {
                int sum =  
                    grid[(i - 1) * N + j - 1] + // [-1, -1]
                    grid[(i - 1) * N + j    ] + // [-1,  0]
                    grid[(i - 1) * N + j + 1] + // [-1,  1]
                    grid[ i      * N + j - 1] + // [ 0, -1]
                    grid[ i      * N + j + 1] + // [ 0,  1]
                    grid[(i + 1) * N + j - 1] + // [ 1, -1]
                    grid[(i + 1) * N + j    ] + // [ 1,  0]
                    grid[(i + 1) * N + j + 1];  // [ 1,  1]
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
        if (t_info.send_first_to != rank) { // if false, there are no other workers
            // copy left and right borders
            for (int row = 1; row < numrows; row++) {
                next_grid[row * N] = next_grid[row * N + N - 2];
                next_grid[row * N + N - 1] = next_grid[row * N + 1];
            }
            // sends the updated portion to other workers
            if (t_info.send_first_to != -1) {
                // sending first row (index 1) to task t_info.send_first_to
                MPI_Send(&next_grid[1 * N], N, MPI_C_BOOL, t_info.send_first_to, 1, MPI_COMM_WORLD);
            }
            if (t_info.send_last_to != -1) {
                // sending last row (index last - 1) to task t_info.send_last_to
                MPI_Send(&next_grid[(numrows - 2) * N], N, MPI_C_BOOL, t_info.send_last_to, 1, MPI_COMM_WORLD);
            }
            // receive upper and lower borders from other tasks
            MPI_Recv(&next_grid[0], N, MPI_C_BOOL, t_info.send_first_to, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&next_grid[(numrows - 1) * N], N, MPI_C_BOOL, t_info.send_last_to, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        else {
            // if there are no workers, functions like sequential version
            copyBorders(next_grid, N);
        }
        swap(&grid, &next_grid);
    }
    // sends the updated portion back to master
    for (int i = 1; i < numrows - 1; i++)
        MPI_Send(&grid[i * N], N, MPI_C_BOOL, 0, 1, MPI_COMM_WORLD);
    free(grid);
    free(next_grid);
}

int main(int argc, char *argv[]) {
    srand(42);
    
    if (argc != 3) {
        printf("gol_sequential.c requires N and G integers.\n");
        return -1;
    }
    int N = atoi(argv[1]) + 2; // size of grid (N x N)
    int G = atoi(argv[2]);     // # of generations

    // MPI INIT
    int result = MPI_Init(NULL, NULL);
    if (result != MPI_SUCCESS) {
        printf("ERROR initializing MPI.\n");
        MPI_Abort(MPI_COMM_WORLD, result);
    }
    int numtasks, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        masterRoutine(numtasks, N, G);
    }
    else {
        workerRoutine(rank, N, G);
    }

    MPI_Finalize();
    return 0;
}
