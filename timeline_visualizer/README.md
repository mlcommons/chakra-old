This tool visualizes the execution timeline of a given execution graph (EG).

You can run this timeline visualizer with the following command.
```
python timeline_visualizer.py\
    --input_filename <input_filename>\
    --output_filename <output_filename>\
    --num_npus <num_npus>\
    --npu_frequency <npu_frequency>
```

The input file is an execution trace file in csv, and the output file is a json file.
The input file format is shown below.
```
issue,<dummy_str>=npu_id,<dummy_str>=curr_cycle,<dummy_str>=node_id,<dummy_str>=node_name
callback,<dummy_str>=npu_id,<dummy_str>=curr_cycle,<dummy_str>=node_id,<dummy_str>=node_name
issue,<dummy_str>=npu_id,<dummy_str>=curr_cycle,<dummy_str>=node_id,<dummy_str>=node_name
issue,<dummy_str>=npu_id,<dummy_str>=curr_cycle,<dummy_str>=node_id,<dummy_str>=node_name
callback,<dummy_str>=npu_id,<dummy_str>=curr_cycle,<dummy_str>=node_id,<dummy_str>=node_name
callback,<dummy_str>=npu_id,<dummy_str>=curr_cycle,<dummy_str>=node_id,<dummy_str>=node_name
...
```
As this tool requires an execution trace of an EG, a simulator has to print out execution traces.
The output json file is chrome-tracing-compatible.
When you open the file with `chrome://tracing`, you will see an execution timeline like the one below.
![](timeline_visualizer.png)
