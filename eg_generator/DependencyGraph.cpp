/*
 * Copyright (c) 2013 - 2015 ARM Limited
 * All rights reserved
 *
 * The license below extends only to copyright in the software and shall
 * not be construed as granting a license to any other intellectual
 * property including but not limited to intellectual property relating
 * to a hardware implementation of the functionality of the software
 * licensed hereunder.  You may use the software subject to the license
 * terms below provided that you ensure that this notice is replicated
 * unmodified and in its entirety in all distributions of the software,
 * modified or unmodified, in source code or in binary form.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met: redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer;
 * redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution;
 * neither the name of the copyright holders nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "DependencyGraph.h"

using namespace std;
using namespace Chakra;

DependencyGraph::DependencyGraph(const string filename)
  : graphStream(nullptr), firstWin(true), depWindowSize(4096), currID(0) {
  graphStream = new ProtoOutputStream(filename);
}

GraphNode* DependencyGraph::addNode(ChakraNode::NodeType node_type) {
  GraphNode* new_node = new GraphNode;
  uint64_t assigned_id = currID++;
  new_node->id = assigned_id;
  new_node->node_type = convertProtobufNodeType2CppNodeType(node_type);
  graphInfoMap[assigned_id] = new_node;
  depTrace.push_back(new_node);
  return new_node;
}

void DependencyGraph::assignDep(
    uint64_t past_node_id, uint64_t new_node_id) {
  auto past_node_iter = graphInfoMap.find(past_node_id);
  auto new_node_iter = graphInfoMap.find(new_node_id);
  assignDep(past_node_iter->second, new_node_iter->second);
}

void DependencyGraph::assignDep(GraphNode* past_node, GraphNode* new_node) {
  new_node->dep_list.push_back(past_node->id);
}

void DependencyGraph::writeTrace(uint32_t num_to_write) {
  depTraceItr dep_graph_itr(depTrace.begin());
  depTraceItr dep_graph_itr_start = dep_graph_itr;
  while (num_to_write > 0) {
    GraphNode* temp_ptr = *dep_graph_itr;

    ProtoMessage::ChakraNode dep_pkt;
    dep_pkt.set_id(temp_ptr->id);
    dep_pkt.set_name(temp_ptr->name);
    dep_pkt.set_node_type(convertCppNodeType2ProtobufNodeType(temp_ptr->node_type));
    while (!temp_ptr->dep_list.empty()) {
      dep_pkt.add_dep(temp_ptr->dep_list.front());
      temp_ptr->dep_list.pop_front();
    }
    dep_pkt.set_tensor_loc(convertCppMemType2ProtobufMemType(temp_ptr->tensor_loc));
    dep_pkt.set_tensor_size(temp_ptr->tensor_size);
    dep_pkt.set_simulated_run_time(temp_ptr->simulated_run_time);
    dep_pkt.set_input_tensor_loc(convertCppMemType2ProtobufMemType(temp_ptr->input_tensor_loc));
    dep_pkt.set_input_tensor_size(temp_ptr->input_tensor_size);
    dep_pkt.set_output_tensor_loc(convertCppMemType2ProtobufMemType(temp_ptr->output_tensor_loc));
    dep_pkt.set_output_tensor_size(temp_ptr->output_tensor_size);
    dep_pkt.set_num_ops(temp_ptr->num_ops);
    dep_pkt.set_comm_type(convertCppCommType2ProtobufCommType(temp_ptr->comm_type));
    for (auto involved_dim: temp_ptr->involved_dim) {
      dep_pkt.add_involved_dim(involved_dim);
    }
    dep_pkt.set_communicator_id(temp_ptr->communicator_id);
    dep_pkt.set_comm_src(temp_ptr->comm_src);
    dep_pkt.set_comm_dst(temp_ptr->comm_dst);
    dep_pkt.set_comm_size(temp_ptr->comm_size);
    dep_pkt.set_comm_tag(temp_ptr->comm_tag);
    dep_pkt.set_comm_priority(temp_ptr->comm_priority);

    graphStream->write(dep_pkt);

    dep_graph_itr++;
    delete temp_ptr;
    num_to_write--;
  }
  depTrace.erase(dep_graph_itr_start, dep_graph_itr);
}

void DependencyGraph::flushEG() {
  // Write to graph all nodes in the depTrace.
  writeTrace(depTrace.size());
  // Delete the stream objects
  delete graphStream;
}

Chakra::GraphNodeType DependencyGraph::convertProtobufNodeType2CppNodeType(
    ChakraNode::NodeType node_type) {
  if (node_type == ChakraNode::INVALID_NODE) {
    return Chakra::GraphNodeType::INVALID_NODE;
  } else if (node_type == ChakraNode::MEM_LOAD_NODE) {
    return Chakra::GraphNodeType::MEM_LOAD_NODE;
  } else if (node_type == ChakraNode::MEM_STORE_NODE) {
    return Chakra::GraphNodeType::MEM_STORE_NODE;
  } else if (node_type == ChakraNode::COMP_NODE) {
    return Chakra::GraphNodeType::COMP_NODE;
  } else if (node_type == ChakraNode::COMM_SEND_NODE) {
    return Chakra::GraphNodeType::COMM_SEND_NODE;
  } else if (node_type == ChakraNode::COMM_RECV_NODE) {
    return Chakra::GraphNodeType::COMM_RECV_NODE;
  } else if (node_type == ChakraNode::COMM_COLL_NODE) {
    return Chakra::GraphNodeType::COMM_COLL_NODE;
  } else {
    assert(false);
  }
  return Chakra::GraphNodeType::INVALID_NODE;
}

ChakraNode::NodeType DependencyGraph::convertCppNodeType2ProtobufNodeType(
    Chakra::GraphNodeType node_type) {
  if (node_type == Chakra::GraphNodeType::INVALID_NODE) {
    return ChakraNode::INVALID_NODE;
  } else if (node_type == Chakra::GraphNodeType::MEM_LOAD_NODE) {
    return ChakraNode::MEM_LOAD_NODE;
  } else if (node_type == Chakra::GraphNodeType::MEM_STORE_NODE) {
    return ChakraNode::MEM_STORE_NODE;
  } else if (node_type == Chakra::GraphNodeType::COMP_NODE) {
    return ChakraNode::COMP_NODE;
  } else if (node_type == Chakra::GraphNodeType::COMM_SEND_NODE) {
    return ChakraNode::COMM_SEND_NODE;
  } else if (node_type == Chakra::GraphNodeType::COMM_RECV_NODE) {
    return ChakraNode::COMM_RECV_NODE;
  } else if (node_type == Chakra::GraphNodeType::COMM_COLL_NODE) {
    return ChakraNode::COMM_COLL_NODE;
  } else {
    assert(false);
  }
  return ChakraNode::INVALID_NODE;
}

Chakra::MemoryType DependencyGraph::convertProtobufMemType2CppMemType(
    ChakraNode::MemoryType mem_type) {
  if (mem_type == ChakraNode::INVALID_MEMORY) {
    return Chakra::MemoryType::INVALID_MEMORY;
  } else if (mem_type == ChakraNode::LOCAL_MEMORY) {
    return Chakra::MemoryType::LOCAL_MEMORY;
  } else if (mem_type == ChakraNode::REMOTE_MEMORY) {
    return Chakra::MemoryType::REMOTE_MEMORY;
  } else {
    assert(false);
  }
  return Chakra::MemoryType::INVALID_MEMORY;
}

ChakraNode::MemoryType DependencyGraph::convertCppMemType2ProtobufMemType(
    Chakra::MemoryType mem_type) {
  if (mem_type == Chakra::MemoryType::INVALID_MEMORY) {
    return ChakraNode::INVALID_MEMORY;
  } else if (mem_type == Chakra::MemoryType::LOCAL_MEMORY) {
    return ChakraNode::LOCAL_MEMORY;
  } else if (mem_type == Chakra::MemoryType::REMOTE_MEMORY) {
    return ChakraNode::REMOTE_MEMORY;
  } else {
    assert(false);
  }
  return ChakraNode::INVALID_MEMORY;
}

Chakra::CollectiveCommType DependencyGraph::convertProtobufCommType2CppCommType(
    ChakraNode::CollectiveCommType comm_type) {
  if (comm_type == ChakraNode::INVALID_COMM) {
    return Chakra::CollectiveCommType::INVALID_COMM;
  } else if (comm_type == ChakraNode::ALL_REDUCE) {
    return Chakra::CollectiveCommType::ALL_REDUCE;
  } else if (comm_type == ChakraNode::ALL_TO_ALL) {
    return Chakra::CollectiveCommType::ALL_TO_ALL;
  } else if (comm_type == ChakraNode::ALL_GATHER) {
    return Chakra::CollectiveCommType::ALL_GATHER;
  } else if (comm_type == ChakraNode::REDUCE_SCATTER) {
    return Chakra::CollectiveCommType::REDUCE_SCATTER;
  } else {
    assert(false);
  }
  return Chakra::CollectiveCommType::INVALID_COMM;
}

ChakraNode::CollectiveCommType DependencyGraph::convertCppCommType2ProtobufCommType(
    Chakra::CollectiveCommType comm_type) {
  if (comm_type == Chakra::CollectiveCommType::INVALID_COMM) {
    return ChakraNode::INVALID_COMM;
  } else if (comm_type == Chakra::CollectiveCommType::ALL_REDUCE) {
    return ChakraNode::ALL_REDUCE;
  } else if (comm_type == Chakra::CollectiveCommType::ALL_TO_ALL) {
    return ChakraNode::ALL_TO_ALL;
  } else if (comm_type == Chakra::CollectiveCommType::ALL_GATHER) {
    return ChakraNode::ALL_GATHER;
  } else if (comm_type == Chakra::CollectiveCommType::REDUCE_SCATTER) {
    return ChakraNode::REDUCE_SCATTER;
  } else {
    assert(false);
  }
  return ChakraNode::INVALID_COMM;
}
