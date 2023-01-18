This is a graph feeder that feeds dependency-free nodes to a simulator.
Therefore, a simulator has to import this feeder as a library.
Currently, ASTRA-sim is the only simulator that supports the graph feeder.
You can run execution graphs on ASTRA-sim with the following commands.
```
$ git clone --recurse-submodules git@github.com:astra-sim/astra-sim.git
$ cd astra-sim
$ git checkout Chakra
$ git submodule update --init --recursive
$ cd extern/graph_frontend/chakra/
$ git checkout main
$ cd -
$ ./build/astra_analytical/build.sh -c

$ cd extern/graph_frontend/chakra/eg_generator
$ cmake CMakeLists.txt && make -j$(nproc)
$ ./eg_generator

$ cd -
$ ./run.sh
```
