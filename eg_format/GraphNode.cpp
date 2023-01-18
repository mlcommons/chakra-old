#include "GraphNode.h"

using namespace std;
using namespace Chakra;

GraphNode::GraphNode()
  : id(0),
  node_type(GraphNodeType::INVALID_NODE),
  tensor_loc(MemoryType::INVALID_MEMORY),
  tensor_size(0),
  simulated_run_time(0),
  input_tensor_loc(MemoryType::INVALID_MEMORY),
  input_tensor_size(0),
  output_tensor_loc(MemoryType::INVALID_MEMORY),
  output_tensor_size(0),
  num_ops(0),
  comm_type(CollectiveCommType::ALL_REDUCE),
  communicator_id(0),
  comm_src(0),
  comm_dst(0),
  comm_size(0),
  comm_tag(0),
  comm_priority(0) {
}

bool GraphNode::removeDep(NodeId dep) {
  for (auto it = dep_list.begin(); it != dep_list.end(); it++) {
    if (*it == dep) {
      dep_list.erase(it);
      return true;
    }
  }
  return false;
}

bool GraphNode::removeDepOn(NodeId node_id) {
  removeDep(node_id);
  return dep_list.empty();
}
