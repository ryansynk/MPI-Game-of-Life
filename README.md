# MPI-Game-of-Life
Simulates Conway's Game of Life in Parallel

Conway's Game of Life (GOL) is an interesting cellular automaton, see [here](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life) for more info. This repo
contains two parallel implementations of GOL using OpenMPI - one using blocking communication, the other using non-blocking. 

To compile, call `make`

To run, call
```
mpirun -np <num_procs> <input_file>.data <num_iterations> <width> <height>
```
Input files contain indices of initially populated squares on the GOL grid. One is provided here in order to run the code. For example, run

```
mpirun -np 4 ./life-blocking final.512x512.data 100 512 512
```

After running, a file of form `<input_file>.<num_iters>.csv` will be generated, which represents the location of pixels on the board after running game of life for `num_iters`.
Additionally, statistics on the average, max, and min runtime across processes will be printed to stdout.
