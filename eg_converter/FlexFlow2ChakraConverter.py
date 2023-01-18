#!/usr/bin/env python3

import copy
import pydot
import sys
sys.path.append("../eg_format/")
sys.path.append("../third_party/utils/")

from execution_graph_def_pb2 import *
from protolib import *

class FlexFlow2ChakraConverter:
    def __init__(self, input_filename, output_filename, npu_frequency, num_dims):
        self.input_filename = input_filename
        self.output_filename = output_filename
        self.num_cycles_per_sec = npu_frequency * 1000 * 1000
        self.num_dims = num_dims
        self.node_id_npu_id_dict = {}
        self.node_id_node_dict = {}

    def get_label(self, ff_node):
        return ff_node.get_attributes()["label"].replace("\"", "")[1:-1]

    def get_id(self, ff_node):
        return int(ff_node.get_name().replace("node", ""))

    def get_npu_id(self, ff_node):
        label = self.get_label(ff_node)
        return int(label.split("|")[0].strip().split("=")[1])

    def get_name(self, ff_node):
        label = self.get_label(ff_node)
        return label.split("|")[1].strip()

    def get_node_type(self, ff_node):
        label = self.get_label(ff_node)
        node_type = label.split("|")[3].strip()
        if node_type == "COMP_NODE":
            return ChakraNode.COMP_NODE
        elif node_type == "COMM_SEND_RECV_NODE":
            return ChakraNode.COMM_SEND_NODE
        else:
            print("{} is unsupported".format(node_type))
            assert False

    def get_simulated_run_time(self, ff_node):
        label = self.get_label(ff_node)
        wall_clock_time = float(label.split("|")[4].strip().split("=")[1])
        return int(round(wall_clock_time * self.num_cycles_per_sec))

    def get_comm_src(self, ff_node):
        label = self.get_label(ff_node)
        return int(label.split("|")[4].strip().split("=")[1])

    def get_comm_dst(self, ff_node):
        label = self.get_label(ff_node)
        return int(label.split("|")[5].strip().split("=")[1])

    def get_comm_size(self, ff_node):
        label = self.get_label(ff_node)
        return int(label.split("|")[6].strip().split("=")[1])

    def convert_FF_node_to_CK_node(self, ff_node):
        ck_node = ChakraNode()
        ck_node.id = self.get_id(ff_node)
        ck_node.name = self.get_name(ff_node)
        ck_node.node_type = self.get_node_type(ff_node)
        if ck_node.node_type == ChakraNode.COMP_NODE:
            ck_node.simulated_run_time = self.get_simulated_run_time(ff_node)
        elif ck_node.node_type == ChakraNode.COMM_SEND_NODE:
            ck_node.comm_src = self.get_comm_src(ff_node)
            ck_node.comm_dst = self.get_comm_dst(ff_node)
            ck_node.comm_size = self.get_comm_size(ff_node)
        self.node_id_npu_id_dict.update({ck_node.id: self.get_npu_id(ff_node)})
        return ck_node

    def convert(self):
        ff_graphs = pydot.graph_from_dot_file(self.input_filename)
        ff_graph = ff_graphs[0]
        assert len(ff_graphs) == 1

        # convert FlexFlow EG to Chakra EG
        npu_ids = set()
        for ff_node in ff_graph.get_nodes():
            ck_node = self.convert_FF_node_to_CK_node(ff_node)
            self.node_id_node_dict.update({ck_node.id: ck_node})
            if ck_node.node_type == ChakraNode.COMP_NODE:
                npu_ids.add(self.node_id_npu_id_dict[ck_node.id])
        for edge in ff_graph.get_edges():
            src_id = int(edge.get_source().replace("node", ""))
            dst_id = int(edge.get_destination().replace("node", ""))
            ck_node = self.node_id_node_dict[dst_id]
            ck_node.dep.append(src_id)

        # generate per-NPU Chakra graphs
        next_comm_tag = 0
        npu_id_node_id_node_dict = {}
        comm_key_comm_tag_dict = {}
        for npu_id in npu_ids:
            npu_id_node_id_node_dict.update({npu_id: {}})
            for node_id in self.node_id_node_dict.keys():
                ck_node = copy.deepcopy(self.node_id_node_dict[node_id])

                # compute nodes
                if ck_node.node_type == ChakraNode.COMP_NODE:
                    ck_node.name = "COMP_NODE_%s" % (ck_node.name)
                    if self.node_id_npu_id_dict[ck_node.id] == npu_id:
                        npu_id_node_id_node_dict[npu_id].update({node_id: ck_node})

                # communication nodes
                elif (ck_node.node_type == ChakraNode.COMM_SEND_NODE):
                    if (ck_node.comm_src == npu_id) or (ck_node.comm_dst == npu_id):
                        comm_key = "%s_%d_%d"\
                                % (ck_node.id, ck_node.comm_src, ck_node.comm_dst)
                        if comm_key not in comm_key_comm_tag_dict.keys():
                            comm_tag = next_comm_tag
                            comm_key_comm_tag_dict.update({comm_key: comm_tag})
                            next_comm_tag += 1
                        else:
                            comm_tag = comm_key_comm_tag_dict[comm_key]

                        # create a new communication node
                        ck_comm_node = ChakraNode()
                        ck_comm_node.id = ck_node.id
                        if ck_node.comm_src == npu_id:
                            ck_comm_node.name = "COMM_SEND_NODE"
                            ck_comm_node.node_type = ChakraNode.COMM_SEND_NODE
                        elif ck_node.comm_dst == npu_id:
                            ck_comm_node.name = "COMM_RECV_NODE"
                            ck_comm_node.node_type = ChakraNode.COMM_RECV_NODE
                        ck_comm_node.name += "_%s" % (ck_node.name)
                        ck_comm_node.comm_src = ck_node.comm_src
                        ck_comm_node.comm_dst = ck_node.comm_dst
                        ck_comm_node.comm_size = ck_node.comm_size
                        ck_comm_node.comm_tag = comm_tag

                        # transfer dependencies
                        for parent_node_id in ck_node.dep:
                            parent_node = self.node_id_node_dict[parent_node_id]
                            if self.node_id_npu_id_dict[parent_node.id] == npu_id:
                                ck_comm_node.dep.append(parent_node_id)

                        npu_id_node_id_node_dict[npu_id].update({node_id: ck_comm_node})

        # write per-NPU Chakra graphs
        for npu_id in sorted(npu_id_node_id_node_dict.keys()):
            filename = self.output_filename + ".%d.eg" % (npu_id)
            with open(filename, "wb") as f:
                for node_id in sorted(npu_id_node_id_node_dict[npu_id].keys()):
                    ck_node = npu_id_node_id_node_dict[npu_id][node_id]
                    encodeMessage(f, ck_node)
