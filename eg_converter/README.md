This tool converts execution graphs into the Chakra format.
This converter supports three types of formats: ASTRA-sim text files, FlexFlow, and PyTorch.
Before running the converter, please make sure that `execution_graph_def_pb2.py` is available.
```
$ cd ../eg_format
$ protoc execution_graph_def.proto --python_out=.
```

You can use the following commands for each input type.

## ASTRA-sim Text Files
```
$ python converter.py\
    --input_type Text\
    --input_filename <input_filename>\
    --output_filename <output_filename>\
    --num_npus <num_npus>\
    --num_dims <num_dims>\
    --num_passes <num_passes>
```

## FlexFlow Execution Graphs
```
$ python converter.py\
    --input_type FlexFlow\
    --input_filename <input_filename>\
    --output_filename <output_filename>\
    --npu_frequency <npu_frequency>\
    --num_dims <num_dims>
```

## PyTorch Execution Graphs
```
$ python converter.py\
    --input_type PyTorch\
    --input_filename <input_filename>\
    --output_filename <output_filename>\
    --default_simulated_run_time <default_simulated_run_time>\
    --num_dims <num_dims>
```
