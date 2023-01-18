This is an execution graph generator that generates synthetic execution graphs.
A user can define a new function in the generator to generate new synthetic execution graphs.
You can follow the commands below to run the generator.
```
$ cd ../eg_format/
$ protoc execution_graph_def.proto --cpp_out=.
$ cd ../eg_generator/
$ cmake CMakeLists.txt && make -j$(nproc)
$ ./eg_generator --num_npus <num_npus> --num_dims <num_dims>
```
