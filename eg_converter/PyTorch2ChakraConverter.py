#!/usr/bin/env python3

import json
import sys
sys.path.append("../eg_format/")
sys.path.append("../third_party/utils/")

from eg_replay_utils import (
        fbgemm_input_args_indices,
        get_input_tensors,
        get_output_tensors,
        has_backward_parent,
        is_backward_aten,
        is_backward_parent,
        is_fbgemm_backward,
        is_fbgemm_forward,
        skip_op
)

from execution_graph import *
from execution_graph_def_pb2 import *
from protolib import *


def is_op(node, strict=False):
    if not strict:
        return node.type == NodeType.OPERATOR
    return node.type == NodeType.OPERATOR and (
        node.parent is not None and node.parent.type != NodeType.OPERATOR
    )


def is_qualified(op):
    return not skip_op(op) and (
        is_backward_aten(op)
        or is_fbgemm_backward(op)
        or (is_op(op, strict=True) and not is_backward_parent(op))
    )


class PyTorch2ChakraConverter:
    def __init__(self, input_filename, output_filename,
            default_simulated_run_time, num_dims):
        with open(input_filename, "r") as f:
            self.exgr = ExecutionGraph(json.load(f))

        self.output_filename = output_filename

        self.default_simulated_run_time = default_simulated_run_time

        self.num_dims = num_dims

        # Nodes/Ops to replay after preprocessing.
        self.sorted_nodes = []

        # Skip the node if their names contain any of the following strings.
        self.skip_node_names = [
            "DataLoader",
            "aten::set_"
        ]

        # This is used to pick out a single iteration when eg trace contains multiple iterations.
        # Basically this label should be captured at the beginning of each iteration so that one iteration
        # is between two consecutive label nodes.
        self.label = ""

        try:
            from param_bench.train.compute.python.tools.fb.internals import (
                add_internal_label,
                add_internal_parallel_nodes_parents,
                add_internal_skip_nodes,
            )
        except ImportError:
            logging.info("FB internals not present")
        else:
            self.skip_node_names = add_internal_skip_nodes(self.skip_node_names)
            self.label = add_internal_label()

        self.operators_count = [0]

    # https://pytorch.org/docs/stable/tensors.html
    # https://github.com/pytorch/pytorch/blob/master/c10/util/Half.h
    def get_data_type_size(self, data_type):
        if data_type == "Tensor(float32)":
            return 4
        elif data_type == "Tensor(float)":
            return 4
        elif data_type == "Tensor(float64)":
            return 8
        elif data_type == "Tensor(double)":
            return 8
        elif data_type == "Tensor(float16)":
            return 2
        elif data_type == "Tensor(half)":
            return 2
        elif data_type == "Tensor(bfloat16)":
            return 2
        elif data_type == "Tensor(complex64)":
            return 8
        elif data_type == "Tensor(complex128)":
            return 16
        elif data_type == "Tensor(uint8)":
            return 1
        elif data_type == "Tensor(int8)":
            return 1
        elif data_type == "Tensor(int16)":
            return 2
        elif data_type == "Tensor(short)":
            return 2
        elif data_type == "Tensor(int32)":
            return 4
        elif data_type == "Tensor(int)":
            return 4
        elif data_type == "Tensor(int64)":
            return 8
        elif data_type == "Tensor(long)":
            return 8
        elif data_type == "Tensor(c10::Half)":
            return 2
        else:
            assert False

    def get_nccl_node(self, node):
        return node.children[0]

    def get_node_type(self, node):
        if node.name == "record_param_comms":
            if len(node.children) == 1:
                return ChakraNode.COMM_COLL_NODE
            else:
                return ChakraNode.INVALID_NODE

        if node.op_schema or node.outputs:
            return ChakraNode.COMP_NODE

        return ChakraNode.INVALID_NODE

    def get_comm_type(self, node):
        nccl_node = self.get_nccl_node(node)
        if nccl_node.name == "nccl:all_reduce":
            return ChakraNode.ALL_REDUCE
        elif nccl_node.name == "nccl:all_to_all":
            return ChakraNode.ALL_TO_ALL
        elif nccl_node.name == "nccl:all_gather":
            return ChakraNode.ALL_GATHER
        elif nccl_node.name == "nccl:reduce_scatter":
            return ChakraNode.REDUCE_SCATTER
        else:
            assert False

    def get_comm_size(self, node):
        nccl_node = self.get_nccl_node(node)
        comm_size = 1
        for input_types in nccl_node.input_types:
            comm_size *= self.get_data_type_size(input_types)
        for input_shape_outer in nccl_node.input_shapes:
            for input_shape_inner in input_shape_outer:
                comm_size = comm_size * input_shape_inner
        return comm_size

    def extract_subgraph(self, root):
        """
        return: all nodes in the subgraph, in the order of node ID (also execution)
        """

        def dfs_traverse(root):
            for child in root.children:
                try:
                    if self.label and self.label in child.name:
                        self.sorted_nodes.append(child)

                    if any(x in child.name for x in self.skip_node_names):
                        continue

                    if is_qualified(child):
                        self.sorted_nodes.append(child)
                    else:
                        dfs_traverse(child)
                except Exception as e:
                    print(f"Graph parse error: {e}, node id: {child.id}")
                    exit(1)

        dfs_traverse(root)
        self.sorted_nodes = sorted(self.sorted_nodes, key=lambda x: x.id)
        for i in range(len(self.sorted_nodes)):
            if self.label and self.label in self.sorted_nodes[i].name:
                self.operators_count.append(i)
        if len(self.operators_count) > 1:
            self.sorted_nodes = self.sorted_nodes[
                self.operators_count[1] + 1 : self.operators_count[2]
            ]
        print("#Operators to execute: ", len(self.sorted_nodes))

    def convert(self):
        nodes = self.exgr.get_nodes(clean=True)
        root = nodes[1]  # 1-base

        self.extract_subgraph(root)

        output_filename = "%s.eg" % (self.output_filename)
        with open(output_filename, "wb") as g:
            prev_node_id = None
            for pt_node in self.sorted_nodes:
                ck_node = ChakraNode()
                ck_node.id = pt_node.id
                if prev_node_id != None:
                    ck_node.dep.append(prev_node_id)
                prev_node_id = pt_node.id
                ck_node.node_type = self.get_node_type(pt_node)
                if ck_node.node_type == ChakraNode.COMP_NODE:
                    ck_node.name = "COMP_NODE_%s" % (pt_node.name)
                    try:
                        ck_node.simulated_run_time = pt_node.dur
                    except:
                        ck_node.simulated_run_time = self.default_simulated_run_time
                elif ck_node.node_type == ChakraNode.COMM_COLL_NODE:
                    ck_node.name = "COMM_COLL_NODE_%s" % (pt_node.children[0].name)
                    ck_node.comm_type = self.get_comm_type(pt_node)
                    ck_node.comm_size = self.get_comm_size(pt_node)
                    for i in range(self.num_dims):
                        ck_node.involved_dim.append(True)

                encodeMessage(g, ck_node)
