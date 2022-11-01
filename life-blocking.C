#include "mpi.h"
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
using namespace std;

/*
 * Simulates Conway's game of life given an input configuration.
 * Simulation is done in parallel using OpenMPI and blocking communication.
 * Run using
 * mpirun -np <num_procs> <input_file>.data <num_iterations> <width> <height>
 * E.g,
 * mpirun -np 4 ./life-blocking final.512x512.data 100 512 512
 *
 * This will output a file called <input_file>.<num_iters>.csv, which represents
 * the location of pixels on the board after running game of life for num_iters
 */

/*
 * Reads the input file line by line and stores it in a 2D matrix.
 */
void read_input_file(int **life, string const &input_file_name) {
    
    // Open the input file for reading.
    ifstream input_file;
    input_file.open(input_file_name);
    if (!input_file.is_open())
        perror("Input file cannot be opened");

    string line, val;
    int x, y;
    while (getline(input_file, line)) {
        stringstream ss(line);
        
        // Read x coordinate.
        getline(ss, val, ',');
        x = stoi(val);
        
        // Read y coordinate.
        getline(ss, val);
        y = stoi(val);

        // Populate the life matrix.
        life[x][y] = 1;
    }
    input_file.close();
}

/* 
 * Writes out the final state of the 2D matrix to a csv file. 
 */
void write_output(int **result_matrix, int X_limit, int Y_limit,
                  string const &input_name, int num_of_generations) {
    
    // Open the output file for writing.
    ofstream output_file;
    string input_file_name = input_name.substr(0, input_name.length() - 5);
    output_file.open(input_file_name + "." + to_string(num_of_generations) +
                    ".csv");
    if (!output_file.is_open())
        perror("Output file cannot be opened");
    
    // Output each live cell on a new line. 
    for (int i = 0; i < X_limit; i++) {
        for (int j = 0; j < Y_limit; j++) {
            if (result_matrix[i][j] == 1) {
                output_file << i << "," << j << "\n";
            }
        }
    }
    output_file.close();
}

/*
 * Processes the life array for the specified number of iterations.
 */
void compute(int **life, int **previous_life, int X_limit, int Y_limit) {
    int neighbors = 0;

    // Update the previous_life matrix with the current life matrix state.
    for (int i = 0; i < X_limit; i++) {
        for (int j = 0; j < Y_limit; j++) {
            previous_life[i + 1][j + 1] = life[i][j];
        }
    }

    // For simulating each generation, calculate the number of live
    // neighbors for each cell and then determine the state of the cell in
    // the next iteration.
    for (int i = 1; i < X_limit + 1; i++) {
        for (int j = 1; j < Y_limit + 1; j++) {
            neighbors = previous_life[i - 1][j - 1] + previous_life[i - 1][j] +
            previous_life[i - 1][j + 1] + previous_life[i][j - 1] +
            previous_life[i][j + 1] + previous_life[i + 1][j - 1] +
            previous_life[i + 1][j] + previous_life[i + 1][j + 1];

            if (previous_life[i][j] == 0) {
                // A cell is born only when an unoccupied cell has 3 neighbors.
                if (neighbors == 3)
                    life[i - 1][j - 1] = 1;
            } else {
                // An occupied cell survives only if it has either 2 or 3 neighbors.
                // The cell dies out of loneliness if its neighbor count is 0 or 1.
                // The cell also dies of overpopulation if its neighbor count is 4-8.
                if (neighbors != 2 && neighbors != 3) {
                    life[i - 1][j - 1] = 0;
                }
            }
        }
    }
}

string datastring(int myrank, int **mylife, int my_X_limit, int Y_limit) {
    string mystr = "";
    mystr = "data for rank = " + to_string(myrank) + "\n";
    for (int j = 0; j < Y_limit; j++) {
        for (int i = 0; i < my_X_limit; i++) {
            mystr += "l[" + to_string(i) + "]" + "[" + to_string(j) + "]" + " = " + to_string(mylife[i][j]) + ", ";
        }
        mystr += "\n";
    }
    return mystr;
}

/**
  * The main function to execute "Game of Life" simulations on a 2D board.
  */
int main(int argc, char *argv[]) {
    int numprocs, myrank;

    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);

    if (argc != 5)
        perror("Expected arguments: ./life <input_file> <num_of_generations> <X_limit> <Y_limit>");

    string input_file_name = argv[1];
    int num_of_generations = stoi(argv[2]);
    int X_limit = stoi(argv[3]);
    int Y_limit = stoi(argv[4]);

    double max, min, avg;

    // Stores sub-block of board for current process.
    int my_X_limit = X_limit / numprocs;
    int **mylife = new int *[my_X_limit];
    for (int i = 0; i < my_X_limit; i++) {
	    mylife[i] = new int[Y_limit];
	    for (int j = 0; j < Y_limit; j++) {
	        mylife[i][j] = 0;
	    }
    }

    // Vectorized form of mylife, used for MPI_Gather call
    int *mylife_vec = new int[my_X_limit * Y_limit];

    // Use my_previous_life to track the previous state of the board.
    // Pad the my_previous_life matrix with 0s on all four sides by setting all
    // cells in the following rows and columns to 0:
    //  1. Row 0
    //  2. Column 0
    //  3. Row my_X_limit+1
    //  4. Column Y_limit+1
    int **my_previous_life = new int *[my_X_limit + 2];
        for (int i = 0; i < my_X_limit+2; i++) {
	    my_previous_life[i] = new int[Y_limit + 2];
	    for (int j = 0; j < Y_limit + 2; j++) {
	        my_previous_life[i][j] = 0;
	    }
    }

    // Declarations, only allocated by rank 0
    int *life_vec;
    int **life;

    if (myrank == 0) {
        // Rank 0 sends board to other processes
        // Vectorized form of life, used for MPI_Gather
        life_vec = new int[X_limit * Y_limit];

	    // Initialize full board
        life = new int *[X_limit];
        for (int i = 0; i < X_limit; i++) {
            life[i] = new int[Y_limit];
            for (int j = 0; j < Y_limit; j++) {
                life[i][j] = 0;
            }
        }

        read_input_file(life, input_file_name);

        // Send board sections to each process
	    int block_num = 0;
	    for (int i = 1; i < numprocs; i++) {
            // determine sub-block of board
	        block_num = i * (X_limit / numprocs);
	        for (int j = 0; j < my_X_limit; j++) {
	            // send each column of sub-block to corresponding process
                MPI_Send(life[block_num + j], Y_limit, MPI_INT, i, 0, MPI_COMM_WORLD);
	        }
	    }

        // Rank 0 doesn't need to send to mylife
	    for (int i = 0; i < my_X_limit; i++) {
	        for (int j = 0; j < Y_limit; j++) {
                    mylife[i][j] = life[i][j];
	        }
	    }
    }

    if (myrank != 0) {
        // Other processes recieve initial sections of board
        for (int i = 0; i < my_X_limit; i++) {
            MPI_Recv(mylife[i], Y_limit, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    double mystarttime, myendtime, mytime;
    mystarttime = MPI_Wtime();
    // computational loop -- iterate over number of generations
    for (int numg = 0; numg < num_of_generations; numg++) {
	    // Send and recieve data from neighbors
	    if (myrank % 2 == 0) {
            // Even processes send first, recieve second
	        if (myrank == 0) {
	    	    // leftmost edge of board -- only needs to send and recieve last column
	            MPI_Send(mylife[my_X_limit - 1], Y_limit, MPI_INT, 
                         myrank+1, 0, MPI_COMM_WORLD);
	    	    MPI_Recv((1 + my_previous_life[my_X_limit + 1]), Y_limit, MPI_INT, 
                         myrank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	        } else {
	            MPI_Send(mylife[0], Y_limit, MPI_INT, myrank-1, 0, MPI_COMM_WORLD);
	            MPI_Send(mylife[my_X_limit - 1], Y_limit, MPI_INT, myrank+1, 
                         0, MPI_COMM_WORLD);

	    	    MPI_Recv((1 + my_previous_life[my_X_limit + 1]), Y_limit, MPI_INT, 
                         myrank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	    	    MPI_Recv((1 + my_previous_life[0]), Y_limit, MPI_INT, myrank-1, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	        }
	    } else {
            // Odd processes recieve first, send second
	        if (myrank == numprocs - 1) {
	    	    // rightmost edge of board -- only needs to send and recieve first column
                MPI_Recv((1 + my_previous_life[0]), Y_limit, MPI_INT, myrank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	    	    MPI_Send(mylife[0], Y_limit, MPI_INT, myrank-1, 0, MPI_COMM_WORLD);
	        } else {
                MPI_Recv((1 + my_previous_life[my_X_limit + 1]), Y_limit, MPI_INT, myrank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv((1 + my_previous_life[0]), Y_limit, MPI_INT, myrank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	    	    MPI_Send(mylife[0], Y_limit, MPI_INT, myrank-1, 0, MPI_COMM_WORLD);
	    	    MPI_Send(mylife[my_X_limit - 1], Y_limit, MPI_INT, myrank+1, 0, MPI_COMM_WORLD);
	        }
	    }
        // Use recieved data to time-step conway
	    compute(mylife, my_previous_life, my_X_limit, Y_limit);
    }
    myendtime = MPI_Wtime();
    mytime = myendtime - mystarttime;

    // Vectorizes my_life into my_life_vec
    for (int i = 0; i < my_X_limit; i++) {
	    for (int j = 0; j < Y_limit; j++) {
	        mylife_vec[i*Y_limit + j] = mylife[i][j];
	    }
    }

    MPI_Gather(mylife_vec, my_X_limit * Y_limit, MPI_INT, life_vec, 
	       my_X_limit * Y_limit, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Reduce(&mytime, &max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&mytime, &min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&mytime, &avg, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (myrank == 0) {
	    // un-vectorizes life_vec into life	
	    for (int i = 0; i < X_limit; i++) {
	        for (int j = 0; j < Y_limit; j++) {
                life[i][j] = life_vec[i*Y_limit + j];
	        }
	    }

        write_output(life, X_limit, Y_limit, input_file_name, num_of_generations);

	    avg = avg / numprocs;
	    cout << "TIME: Min: " << min << " s Avg: " << avg << " s Max: " << max << " s\n";

        for (int i = 0; i < X_limit; i++) {
            delete life[i];
        }
        delete[] life;
        delete[] life_vec;
    }

    for (int i = 0; i < my_X_limit; i++) {
        delete mylife[i];
    }
    for (int i = 0; i < my_X_limit+2; i++) {
        delete my_previous_life[i];
    }
    delete[] mylife;
    delete[] my_previous_life;
    delete[] mylife_vec;

    MPI_Finalize();
    return 0;
}
