This tool visualizes a given execution graph (EG) by converting the EG to a graphviz EG.
A user has to feed the output graphviz file to a graphviz visualizer such as https://dreampuf.github.io/GraphvizOnline/.

You can run this tool with the following command.
```
$ cd ../eg_format/
$ protoc execution_graph_def.proto --python_out=`pwd`
$ cd ../eg_visualizer/
$ python eg_visualizer.py\
    --input_filename <input_filename>\
    --output_filename <output_filename>
```
