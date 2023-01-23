#!/usr/bin/env python3

import sys
sys.path.append("../eg_format/")
sys.path.append("../third_party/utils/")

from execution_graph_def_pb2 import *
from protolib import *

class Layer:
    def __init__(self, line):
        col = line.strip().split()
        self.name = col[0]

        # forward
        self.fwd_comp_time = int(col[2])
        self.fwd_comm_type = str(col[3])
        self.fwd_comm_size = int(col[4])
        self.fwd_comp_node = None
        self.fwd_comm_node = None

        # backward input gradient
        self.bwd_ig_comp_time = int(col[5])
        self.bwd_ig_comm_type = str(col[6])
        self.bwd_ig_comm_size = int(col[7])
        self.bwd_ig_comp_node = None
        self.bwd_ig_comm_node = None

        # backward weight gradient
        self.bwd_wg_comp_time = int(col[8])
        self.bwd_wg_comm_type = str(col[9])
        self.bwd_wg_comm_size = int(col[10])
        self.bwd_wg_update_time = str(col[11])
        self.bwd_wg_comp_node = None
        self.bwd_wg_comm_node = None

class Text2ChakraConverter:
    def __init__(self, input_filename, output_filename, num_npus, num_dims,
            num_passes):
        self.input_filename = input_filename
        self.output_filename = output_filename
        self.num_npus = num_npus
        self.num_dims = num_dims
        self.num_passes = num_passes
        self.next_node_id = 0

    def get_layers(self, f, num_layers):
        layers = []
        for line in f:
            layers.append(Layer(line))
        return layers

    def get_node(self, name, node_type):
        node = ChakraNode()
        node.id = self.next_node_id
        self.next_node_id += 1
        node.name = name
        node.node_type = node_type
        return node

    def get_comp_node(self, layer_name, phase, comp_time):
        node = self.get_node("COMP_NODE_" + layer_name + "_" + phase,
                ChakraNode.COMP_NODE)
        node.simulated_run_time = comp_time
        return node

    def get_comm_type(self, comm_type):
        if comm_type == "ALLREDUCE":
            return ChakraNode.ALL_REDUCE
        elif comm_type == "ALLTOALL":
            return ChakraNode.ALL_TO_ALL
        elif comm_type == "ALLGATHER":
            return ChakraNode.ALL_GATHER
        elif comm_type == "REDUCESCATTER":
            return ChakraNode.REDUCE_SCATTER

    def get_coll_comm_node(self, layer_name, comm_type, comm_size):
        node = self.get_node(
                "COMM_COLL_NODE_" + layer_name + "_" + comm_type,
                ChakraNode.COMM_COLL_NODE)
        node.comm_type = self.get_comm_type(comm_type)
        node.comm_size = comm_size
        return node

    def add_dep(self, child_node, parent_node):
        child_node.dep.append(parent_node.id)

    def convert(self):
        with open(self.input_filename, "r") as f:
            first_line = f.readline().strip().split()
            parallelism_type = first_line[0]
            num_layers = int(f.readline().strip())

            if parallelism_type == "MICRO":
                self.convert_microbenchmark(f, num_layers)
            elif parallelism_type == "DATA":
                self.convert_data_parallel(f, num_layers)
            elif parallelism_type == "MODEL":
                self.convert_model_parallel(f, num_layers)
            elif (parallelism_type == "HYBRID_DATA_MODEL"):
                self.convert_hybrid_data_model(f, num_layers)
            elif (parallelism_type == "HYBRID_MODEL_DATA"):
                self.convert_hybrid_model_data(f, num_layers)
            elif (parallelism_type == "HYBRID_DLRM")\
                    or (parallelism_type == "HYBRID_DLRM_ENHANCED"):
                last_bottom_layer = int(first_line[1])
                self.convert_hybrid_dlrm(f, num_layers, last_bottom_layer)
            elif parallelism_type == "HYBRID_TRANSFORMER":
                options = " ".join(first_line[1:])
                self.convert_hybrid_transformer(f, num_layers, options)
            else:
                print("Unsupported parallelism type")

    def convert_microbenchmark(self, f, num_layers):
        layers = self.get_layers(f, num_layers)
        for npu_id in range(self.num_npus):
            output_filename = "%s.%d.eg" % (self.output_filename, npu_id)
            with open(output_filename, "wb") as g:
                for i in range(self.num_passes):
                    for layer in layers:
                        bwd_wg_comm_node = self.get_coll_comm_node(
                                layer.name,
                                layer.bwd_wg_comm_type,
                                layer.bwd_wg_comm_size)
                        for j in range(self.num_dims):
                            bwd_wg_comm_node.involved_dim.append(True)
                        encodeMessage(g, bwd_wg_comm_node)

    def convert_data_parallel(self, f, num_layers):
        layers = self.get_layers(f, num_layers)
        for npu_id in range(self.num_npus):
            output_filename = "%s.%d.eg" % (self.output_filename, npu_id)
            with open(output_filename, "wb") as g:
                for i in range(self.num_passes):
                    # forward pass
                    for idx, layer in enumerate(layers):
                        fwd_comp_node = self.get_comp_node(
                                layer.name, "FWD",
                                layer.fwd_comp_time)
                        if idx != 0:
                            self.add_dep(fwd_comp_node, layers[idx-1].fwd_comp_node)
                        if layer.bwd_wg_comm_node != None:
                            self.add_dep(fwd_comp_node, layer.bwd_wg_comm_node)
                        layer.fwd_comp_node = fwd_comp_node
                        encodeMessage(g, fwd_comp_node)

                    # backward pass
                    for idx, layer in enumerate(reversed(layers)):
                        bwd_wg_comp_node = self.get_comp_node(
                                layer.name, "BWD_WG",
                                layer.bwd_wg_comp_time)
                        if idx == 0:
                            self.add_dep(bwd_wg_comp_node, fwd_comp_node)
                        else:
                            self.add_dep(bwd_wg_comp_node,
                                    layers[len(layers)-idx].bwd_ig_comp_node)
                        encodeMessage(g, bwd_wg_comp_node)

                        bwd_wg_comm_node = self.get_coll_comm_node(
                                layer.name,
                                layer.bwd_wg_comm_type,
                                layer.bwd_wg_comm_size)
                        for j in range(self.num_dims):
                            bwd_wg_comm_node.involved_dim.append(True)
                        self.add_dep(bwd_wg_comm_node, bwd_wg_comp_node)
                        layer.bwd_wg_comm_node = bwd_wg_comm_node
                        encodeMessage(g, bwd_wg_comm_node)

                        if idx != (len(layers) - 1):
                            bwd_ig_comp_node = self.get_comp_node(
                                    layer.name, "BWD_IG",
                                    layer.bwd_ig_comp_time)
                            self.add_dep(bwd_ig_comp_node, bwd_wg_comp_node)
                            layer.bwd_ig_comp_node = bwd_ig_comp_node
                            encodeMessage(g, bwd_ig_comp_node)

    def convert_model_parallel(self, f, num_layers):
        layers = self.get_layers(f, num_layers)
        for npu_id in range(self.num_npus):
            output_filename = "%s.%d.eg" % (self.output_filename, npu_id)
            with open(output_filename, "wb") as g:
                for i in range(self.num_passes):
                    # forward pass
                    for idx, layer in enumerate(layers):
                        fwd_comp_node = self.get_comp_node(
                                layer.name, "FWD",
                                layer.fwd_comp_time)
                        if idx != 0:
                            self.add_dep(fwd_comp_node, layers[idx-1].fwd_comm_node)
                        if layer.bwd_wg_comp_node != None:
                            self.add_dep(fwd_comp_node, layer.bwd_wg_comp_node)
                        layer.fwd_comp_node = fwd_comp_node
                        encodeMessage(g, fwd_comp_node)

                        fwd_comm_node = self.get_coll_comm_node(
                                layer.name,
                                layer.fwd_comm_type,
                                layer.fwd_comm_size)
                        for j in range(self.num_dims):
                            fwd_comm_node.involved_dim.append(True)
                        layer.fwd_comm_node = fwd_comm_node
                        self.add_dep(fwd_comm_node, fwd_comp_node)
                        encodeMessage(g, fwd_comm_node)

                    # backward pass
                    for idx, layer in enumerate(reversed(layers)):
                        bwd_ig_comp_node = self.get_comp_node(
                                layer.name, "BWD_IG",
                                layer.bwd_ig_comp_time)
                        if idx == 0:
                            self.add_dep(bwd_ig_comp_node, fwd_comm_node)
                        else:
                            self.add_dep(bwd_ig_comp_node,
                                    layers[len(layers)-idx].bwd_wg_comp_node)
                            self.add_dep(bwd_ig_comp_node,
                                    layers[len(layers)-idx].bwd_ig_comm_node)
                        encodeMessage(g, bwd_ig_comp_node)

                        if idx != (num_layers - 1):
                            bwd_ig_comm_node = self.get_coll_comm_node(
                                    layer.name,
                                    layer.bwd_ig_comm_type,
                                    layer.bwd_ig_comm_size)
                            for j in range(self.num_dims):
                                bwd_ig_comm_node.involved_dim.append(True)
                            self.add_dep(bwd_ig_comm_node, bwd_ig_comp_node)
                            layer.bwd_ig_comm_node = bwd_ig_comm_node
                            encodeMessage(g, bwd_ig_comm_node)

                        bwd_wg_comp_node = self.get_comp_node(
                                layer.name, "BWD_WG",
                                layer.bwd_wg_comp_time)
                        self.add_dep(bwd_wg_comp_node, bwd_ig_comp_node)
                        layer.bwd_wg_comp_node = bwd_wg_comp_node
                        encodeMessage(g, bwd_wg_comp_node)

    def convert_hybrid_data_model(self, f, num_layers):
        layers = self.get_layers(f, num_layers)
        for npu_id in range(self.num_npus):
            output_filename = "%s.%d.eg" % (self.output_filename, npu_id)
            with open(output_filename, "wb") as g:
                for i in range(self.num_passes):
                    # forward pass
                    for idx, layer in enumerate(layers):
                        fwd_comp_node = self.get_comp_node(
                                layer.name, "FWD",
                                layer.fwd_comp_time)
                        if layer.bwd_wg_comm_node != None:
                            self.add_dep(fwd_comp_node, layer.bwd_wg_comm_node)
                        if idx != 0:
                            self.add_dep(fwd_comp_node, layers[idx-1].fwd_comm_node)
                        encodeMessage(g, fwd_comp_node)

                        fwd_comm_node = self.get_coll_comm_node(
                                layer.name,
                                layer.fwd_comm_type,
                                layer.fwd_comm_size)
                        fwd_comm_node.involved_dim.append(True)
                        for j in range(self.num_dims - 1):
                            fwd_comm_node.involved_dim.append(False)
                        self.add_dep(fwd_comm_node, fwd_comp_node)
                        layer.fwd_comm_node = fwd_comm_node
                        encodeMessage(g, fwd_comm_node)

                    # backward pass
                    for idx, layer in enumerate(reversed(layers)):
                        bwd_ig_comp_node = self.get_comp_node(
                                layer.name, "BWD_IG",
                                layer.bwd_ig_comp_time)
                        if idx == 0:
                            self.add_dep(bwd_ig_comp_node, fwd_comm_node)
                        else:
                            self.add_dep(bwd_ig_comp_node,
                                    layers[len(layers)-idx].bwd_wg_comp_node)
                            self.add_dep(bwd_ig_comp_node,
                                    layers[len(layers)-idx].bwd_ig_comm_node)
                        encodeMessage(g, bwd_ig_comp_node)

                        if idx != num_layers - 1:
                            bwd_ig_comm_node = self.get_coll_comm_node(
                                    layer.name + "_IG_COMM_",
                                    layer.bwd_ig_comm_type,
                                    layer.bwd_ig_comm_size)
                            bwd_ig_comm_node.involved_dim.append(True)
                            for j in range(self.num_dims - 1):
                                bwd_ig_comm_node.involved_dim.append(False)
                            self.add_dep(bwd_ig_comm_node, bwd_ig_comp_node)
                            layer.bwd_ig_comm_node = bwd_ig_comm_node
                            encodeMessage(g, bwd_ig_comm_node)

                        bwd_wg_comp_node = self.get_comp_node(
                                layer.name, "BWD_WG",
                                layer.bwd_wg_comp_time)
                        self.add_dep(bwd_wg_comp_node, bwd_ig_comp_node)
                        layer.bwd_wg_comp_node = bwd_wg_comp_node
                        encodeMessage(g, bwd_wg_comp_node)

                        bwd_wg_comm_node = self.get_coll_comm_node(
                                layer.name,
                                layer.bwd_wg_comm_type,
                                layer.bwd_wg_comm_size)
                        bwd_wg_comm_node.involved_dim.append(False)
                        for j in range(self.num_dims - 1):
                            bwd_wg_comm_node.involved_dim.append(True)
                        self.add_dep(bwd_wg_comm_node, bwd_wg_comp_node)
                        layer.bwd_wg_comm_node = bwd_wg_comm_node
                        encodeMessage(g, bwd_wg_comm_node)

    def convert_hybrid_model_data(self, f, num_layers):
        layers = self.get_layers(f, num_layers)
        for npu_id in range(self.num_npus):
            output_filename = "%s.%d.eg" % (self.output_filename, npu_id)
            with open(output_filename, "wb") as g:
                for i in range(self.num_passes):
                    # forward pass
                    for idx, layer in enumerate(layers):
                        fwd_comp_node = self.get_comp_node(
                                layer.name, "FWD",
                                layer.fwd_comp_time)
                        if layer.bwd_wg_comm_node != None:
                            self.add_dep(fwd_comp_node, layer.bwd_wg_comm_node)
                        if idx != 0:
                            self.add_dep(fwd_comp_node, layers[idx-1].fwd_comm_node)
                        encodeMessage(g, fwd_comp_node)

                        fwd_comm_node = self.get_coll_comm_node(
                                layer.name,
                                layer.fwd_comm_type,
                                layer.fwd_comm_size)
                        fwd_comm_node.involved_dim.append(False)
                        for j in range(self.num_dims - 1):
                            fwd_comm_node.involved_dim.append(True)
                        self.add_dep(fwd_comm_node, fwd_comp_node)
                        layer.fwd_comm_node = fwd_comm_node
                        encodeMessage(g, fwd_comm_node)

                    # backward pass
                    for idx, layer in enumerate(reversed(layers)):
                        bwd_ig_comp_node = self.get_comp_node(
                                layer.name, "BWD_IG",
                                layer.bwd_ig_comp_time)
                        if idx == 0:
                            self.add_dep(bwd_ig_comp_node, fwd_comm_node)
                        else:
                            self.add_dep(bwd_ig_comp_node, layers[len(layers)-idx].bwd_wg_comp_node)
                            self.add_dep(bwd_ig_comp_node, layers[len(layers)-idx].bwd_ig_comm_node)
                        encodeMessage(g, bwd_ig_comp_node)

                        if idx != num_layers - 1:
                            bwd_ig_comm_node = self.get_coll_comm_node(
                                    layer.name,
                                    layer.bwd_ig_comm_type,
                                    layer.bwd_ig_comm_size)
                            bwd_ig_comm_node.involved_dim.append(False)
                            for j in range(self.num_dims - 1):
                                bwd_ig_comm_node.involved_dim.append(True)
                            self.add_dep(bwd_ig_comm_node, bwd_ig_comp_node)
                            layer.bwd_ig_comm_node = bwd_ig_comm_node
                            encodeMessage(g, bwd_ig_comm_node)

                        bwd_wg_comp_node = self.get_comp_node(
                                layer.name, "BWD_WG",
                                layer.bwd_wg_comp_time)
                        self.add_dep(bwd_wg_comp_node, bwd_ig_comp_node)
                        layer.bwd_wg_comp_node = bwd_wg_comp_node
                        encodeMessage(g, bwd_wg_comp_node)

                        bwd_wg_comm_node = self.get_coll_comm_node(
                                layer.name,
                                layer.bwd_wg_comm_type,
                                layer.bwd_wg_comm_size)
                        bwd_wg_comm_node.involved_dim.append(True)
                        for j in range(self.num_dims - 1):
                            bwd_wg_comm_node.involved_dim.append(False)
                        self.add_dep(bwd_wg_comm_node, bwd_wg_comp_node)
                        layer.bwd_wg_comm_node = bwd_wg_comm_node
                        encodeMessage(g, bwd_wg_comm_node)

    def convert_hybrid_dlrm(self, f, num_layers, last_bottom_layer):
        layers = self.get_layers(f, num_layers)
        for npu_id in range(self.num_npus):
            output_filename = "%s.%d.eg" % (self.output_filename, npu_id)
            with open(output_filename, "wb") as g:
                for i in range(self.num_passes):
                    # forward pass
                    for idx, layer in enumerate(layers):
                        fwd_comp_node = self.get_comp_node(
                                layer.name, "FWD",
                                layer.fwd_comp_time)
                        if layer.bwd_wg_comm_node != None:
                            self.add_dep(fwd_comp_node, layer.bwd_wg_comm_node)
                        elif layer.bwd_wg_comp_node != None:
                            self.add_dep(fwd_comp_node, layer.bwd_wg_comp_node)
                        if idx != 0:
                            self.add_dep(fwd_comp_node, layers[idx-1].fwd_comp_node)
                        if idx == last_bottom_layer:
                            self.add_dep(fwd_comp_node, layers[0].fwd_comm_node)
                        layer.fwd_comp_node = fwd_comp_node
                        encodeMessage(g, fwd_comp_node)

                        if layer.fwd_comm_type == "ALLTOALL":
                            fwd_comm_node = self.get_coll_comm_node(
                                    layer.name,
                                    layer.fwd_comm_type,
                                    layer.fwd_comm_size)
                            for j in range(self.num_dims):
                                fwd_comm_node.involved_dim.append(True)
                            self.add_dep(fwd_comm_node, fwd_comp_node)
                            layer.fwd_comm_node = fwd_comm_node
                            encodeMessage(g, fwd_comm_node)

                    # backward pass
                    for idx, layer in enumerate(reversed(layers)):
                        bwd_wg_comp_node = self.get_comp_node(
                                layer.name, "BWD_WG",
                                layer.bwd_wg_comp_time)
                        if idx == 0:
                            self.add_dep(bwd_wg_comp_node, fwd_comp_node)
                        else:
                            if layers[len(layers)-idx].bwd_ig_comp_node != None:
                                self.add_dep(bwd_wg_comp_node,
                                        layers[len(layers)-idx].bwd_ig_comp_node)
                            if layers[len(layers)-idx-1].bwd_ig_comm_node != None:
                                self.add_dep(bwd_wg_comp_node,
                                        layers[len(layers)-idx-1].bwd_ig_comm_node)
                        layer.bwd_wg_comp_node = bwd_wg_comp_node
                        encodeMessage(g, bwd_wg_comp_node)

                        if layer.bwd_wg_comm_type != "NONE":
                            bwd_wg_comm_node = self.get_coll_comm_node(
                                    layer.name,
                                    layer.bwd_wg_comm_type,
                                    layer.bwd_wg_comm_size)
                            for j in range(self.num_dims):
                                bwd_wg_comm_node.involved_dim.append(True)
                            self.add_dep(bwd_wg_comm_node, bwd_wg_comp_node)
                            layer.bwd_wg_comm_node = bwd_wg_comm_node
                            encodeMessage(g, bwd_wg_comm_node)

                        if idx != (len(layers) - 1):
                            bwd_ig_comp_node = self.get_comp_node(
                                    layer.name, "BWD_IG",
                                    layer.bwd_ig_comp_time)
                            self.add_dep(bwd_ig_comp_node, bwd_wg_comp_node)
                            layer.bwd_ig_comp_node = bwd_ig_comp_node
                            encodeMessage(g, bwd_ig_comp_node)

                        if (len(layers) - idx - 1) == (last_bottom_layer + 1):
                            bwd_ig_comm_node = self.get_coll_comm_node(
                                    layers[0].name,
                                    layers[0].bwd_ig_comm_type,
                                    layers[0].bwd_ig_comm_size)
                            for j in range(self.num_dims):
                                bwd_ig_comm_node.involved_dim.append(True)
                            self.add_dep(bwd_ig_comm_node, bwd_ig_comp_node)
                            layers[0].bwd_ig_comm_node = bwd_ig_comm_node
                            encodeMessage(g, bwd_ig_comm_node)

    def convert_hybrid_transformer(self, f, num_layers, options):
        def break_dimension(model_parallel_npu_group):
            # TODO
            return 0

        layers = self.get_layers(f, num_layers)
        model_parallel_npu_group = int(options.replace("model_parallel_NPU_group: ", ""))
        model_parallel_boundary = break_dimension(model_parallel_npu_group)
        for npu_id in range(self.num_npus):
            output_filename = "%s.%d.eg" % (self.output_filename, npu_id)
            with open(output_filename, "wb") as g:
                for i in range(self.num_passes):
                    # forward pass
                    for idx, layer in enumerate(layers):
                        fwd_comp_node = self.get_comp_node(
                                layer.name, "FWD",
                                layer.fwd_comp_time)
                        if idx == 0:
                            if layer.bwd_wg_comm_node != None:
                                self.add_dep(fwd_comp_node, layer.bwd_wg_comm_node)
                            elif layer.bwd_wg_comp_node != None:
                                self.add_dep(fwd_comp_node, layer.bwd_wg_comp_node)
                        else:
                            if layers[idx-1].fwd_comm_node == None:
                                self.add_dep(fwd_comp_node, layers[idx-1].fwd_comp_node)
                            else:
                                self.add_dep(fwd_comp_node, layers[idx-1].fwd_comm_node)
                            if layer.bwd_wg_comm_node != None:
                                self.add_dep(fwd_comp_node, layer.bwd_wg_comm_node)
                            elif layer.bwd_wg_comp_node != None:
                                self.add_dep(fwd_comp_node, layer.bwd_wg_comp_node)
                        layer.fwd_comp_node = fwd_comp_node
                        encodeMessage(g, fwd_comp_node)

                        fwd_comm_node = None
                        if layer.fwd_comm_type != "NONE":
                            fwd_comm_node = self.get_coll_comm_node(
                                    layer.name,
                                    layer.fwd_comm_type,
                                    layer.fwd_comm_size)
                            for j in range(model_parallel_boundary):
                                fwd_comm_node.involved_dim.append(True)
                            for j in range(self.num_dims - model_parallel_boundary):
                                fwd_comm_node.involved_dim.append(False)
                            layer.fwd_comm_node = fwd_comm_node
                            self.add_dep(fwd_comm_node, fwd_comp_node)
                            encodeMessage(g, fwd_comm_node)

                    # backward pass
                    for idx, layer in enumerate(reversed(layers)):
                        bwd_ig_comp_node = self.get_comp_node(
                                layer.name, "BWD_IG",
                                layer.bwd_ig_comp_time)
                        if idx == 0:
                            if fwd_comm_node == None:
                                self.add_dep(bwd_ig_comp_node, fwd_comp_node)
                            else:
                                self.add_dep(bwd_ig_comp_node, fwd_comm_node)
                        else:
                            self.add_dep(bwd_ig_comp_node,
                                    layers[len(layers)-idx].bwd_wg_comp_node)
                            if layers[len(layers)-idx].bwd_ig_comm_node == None:
                                self.add_dep(bwd_ig_comp_node,
                                        layers[len(layers)-idx].bwd_ig_comp_node)
                            else:
                                self.add_dep(bwd_ig_comp_node,
                                        layers[len(layers)-idx].bwd_ig_comm_node)
                        layer.bwd_ig_comp_node = bwd_ig_comp_node
                        encodeMessage(g, bwd_ig_comp_node)

                        if layer.bwd_ig_comm_type != "NONE":
                            bwd_ig_comm_node = self.get_coll_comm_node(
                                    layer.name,
                                    layer.bwd_ig_comm_type,
                                    layer.bwd_ig_comm_size)
                            for j in range(model_parallel_boundary):
                                bwd_ig_comm_node.involved_dim.append(True)
                            for j in range(self.num_dims - model_parallel_boundary):
                                bwd_ig_comm_node.involved_dim.append(False)
                            self.add_dep(bwd_ig_comm_node, bwd_ig_comp_node)
                            layer.bwd_ig_comm_node = bwd_ig_comm_node
                            encodeMessage(g, bwd_ig_comm_node)

                        bwd_wg_comp_node = self.get_comp_node(
                                layer.name, "BWD_WG",
                                layer.bwd_wg_comp_time)
                        self.add_dep(bwd_wg_comp_node, bwd_ig_comp_node)
                        layer.bwd_wg_comp_node = bwd_wg_comp_node
                        encodeMessage(g, bwd_wg_comp_node)

                        if layer.bwd_wg_comm_type != "NONE":
                            bwd_wg_comm_node = self.get_coll_comm_node(
                                    layer.name,
                                    layer.bwd_wg_comm_type,
                                    layer.bwd_wg_comm_size)
                            for j in range(model_parallel_boundary):
                                bwd_wg_comm_node.involved_dim.append(False)
                            for j in range(self.num_dims - model_parallel_boundary):
                                bwd_wg_comm_node.involved_dim.append(True)
                            self.add_dep(bwd_wg_comm_node, bwd_wg_comp_node)
                            layer.bwd_wg_comm_node = bwd_wg_comm_node
                            encodeMessage(g, bwd_wg_comm_node)
