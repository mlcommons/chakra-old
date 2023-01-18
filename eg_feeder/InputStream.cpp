#include "eg_feeder/InputStream.h"

#include <cassert>

using namespace std;
using namespace Chakra;

typedef ProtoMessage::ChakraNode ChakraNode;

InputStream::InputStream(const string& filename)
  : trace(filename) {
}

void InputStream::reset() {
  trace.reset();
}

bool InputStream::read(GraphNode* element) {
  ProtoMessage::ChakraNode pkt_msg;
  if (trace.read(pkt_msg)) {
    element->id = pkt_msg.id();
    element->name = pkt_msg.name();
    if (pkt_msg.node_type() == ChakraNode::INVALID_NODE) {
      element->node_type = GraphNodeType::INVALID_NODE;
    } else if (pkt_msg.node_type() == ChakraNode::MEM_LOAD_NODE) {
      element->node_type = GraphNodeType::MEM_LOAD_NODE;
    } else if (pkt_msg.node_type() == ChakraNode::MEM_STORE_NODE) {
      element->node_type = GraphNodeType::MEM_STORE_NODE;
    } else if (pkt_msg.node_type() == ChakraNode::COMP_NODE) {
      element->node_type = GraphNodeType::COMP_NODE;
    } else if (pkt_msg.node_type() == ChakraNode::COMM_SEND_NODE) {
      element->node_type = GraphNodeType::COMM_SEND_NODE;
    } else if (pkt_msg.node_type() == ChakraNode::COMM_RECV_NODE) {
      element->node_type = GraphNodeType::COMM_RECV_NODE;
    } else if (pkt_msg.node_type() == ChakraNode::COMM_COLL_NODE) {
      element->node_type = GraphNodeType::COMM_COLL_NODE;
    } else {
      assert(false);
    }
    element->dep_list.clear();
    for (int i = 0; i < (pkt_msg.dep()).size(); i++) {
      element->dep_list.push_back(pkt_msg.dep(i));
    }
    if (pkt_msg.tensor_loc() == ChakraNode::INVALID_MEMORY) {
      element->tensor_loc = MemoryType::INVALID_MEMORY;
    } else if (pkt_msg.tensor_loc() == ChakraNode::LOCAL_MEMORY) {
      element->tensor_loc = MemoryType::LOCAL_MEMORY;
    } else if (pkt_msg.tensor_loc() == ChakraNode::REMOTE_MEMORY) {
      element->tensor_loc = MemoryType::REMOTE_MEMORY;
    } else {
      assert(false);
    }
    element->tensor_size = pkt_msg.tensor_size();
    element->simulated_run_time = pkt_msg.simulated_run_time();
    if (pkt_msg.input_tensor_loc() == ChakraNode::INVALID_MEMORY) {
      element->input_tensor_loc = MemoryType::INVALID_MEMORY;
    } else if (pkt_msg.input_tensor_loc() == ChakraNode::LOCAL_MEMORY) {
      element->input_tensor_loc = MemoryType::LOCAL_MEMORY;
    } else if (pkt_msg.input_tensor_loc() == ChakraNode::REMOTE_MEMORY) {
      element->input_tensor_loc = MemoryType::REMOTE_MEMORY;
    } else {
      assert(false);
    }
    element->input_tensor_size = pkt_msg.input_tensor_size();
    if (pkt_msg.output_tensor_loc() == ChakraNode::INVALID_MEMORY) {
      element->output_tensor_loc = MemoryType::INVALID_MEMORY;
    } else if (pkt_msg.output_tensor_loc() == ChakraNode::LOCAL_MEMORY) {
      element->output_tensor_loc = MemoryType::LOCAL_MEMORY;
    } else if (pkt_msg.output_tensor_loc() == ChakraNode::REMOTE_MEMORY) {
      element->output_tensor_loc = MemoryType::REMOTE_MEMORY;
    }
    element->output_tensor_size = pkt_msg.output_tensor_size();
    element->num_ops = pkt_msg.num_ops();
    if (pkt_msg.comm_type() == ChakraNode::INVALID_COMM) {
      element->comm_type = CollectiveCommType::INVALID_COMM;
    } else if (pkt_msg.comm_type() == ChakraNode::ALL_REDUCE) {
      element->comm_type = CollectiveCommType::ALL_REDUCE;
    } else if (pkt_msg.comm_type() == ChakraNode::ALL_TO_ALL) {
      element->comm_type = CollectiveCommType::ALL_TO_ALL;
    } else if (pkt_msg.comm_type() == ChakraNode::ALL_GATHER) {
      element->comm_type = CollectiveCommType::ALL_GATHER;
    } else if (pkt_msg.comm_type() == ChakraNode::REDUCE_SCATTER) {
      element->comm_type = CollectiveCommType::REDUCE_SCATTER;
    } else {
      assert(false);
    }
    element->involved_dim.clear();
    for (int i = 0; i < (pkt_msg.involved_dim()).size(); i++) {
      element->involved_dim.push_back(pkt_msg.involved_dim(i));
    }
    element->communicator_id = pkt_msg.communicator_id();
    element->comm_src = pkt_msg.comm_src();
    element->comm_dst = pkt_msg.comm_dst();
    element->comm_size = pkt_msg.comm_size();
    element->comm_tag = pkt_msg.comm_tag();
    element->comm_priority = pkt_msg.comm_priority();

    return true;
  }

  // We have reached the end of the file
  return false;
}
