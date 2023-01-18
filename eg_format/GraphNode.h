#pragma once

#include <list>
#include <string>
#include <vector>

#include "eg_format/execution_graph_def.pb.h"
#include "third_party/utils/protoio.hh"

namespace Chakra {

typedef uint64_t NodeId;

enum GraphNodeType
{
  INVALID_NODE = 0,
  MEM_LOAD_NODE,
  MEM_STORE_NODE,
  COMP_NODE,
  COMM_SEND_NODE,
  COMM_RECV_NODE,
  COMM_COLL_NODE
};

enum MemoryType
{
  INVALID_MEMORY = 0,
  LOCAL_MEMORY,
  REMOTE_MEMORY
};

enum CollectiveCommType
{
  INVALID_COMM = 0,
  ALL_REDUCE,
  ALL_TO_ALL,
  ALL_GATHER,
  REDUCE_SCATTER
};

class GraphNode {
 public:
  typedef std::list<NodeId> DepList;

  GraphNode();
  bool removeDep(NodeId dep);
  bool removeDepOn(NodeId node_id);

  NodeId id;
  std::string name;
  GraphNodeType node_type;
  GraphNode::DepList dep_list;
  std::vector<GraphNode*> dependents;
  MemoryType tensor_loc;
  uint64_t tensor_size;
  uint64_t simulated_run_time;
  MemoryType input_tensor_loc;
  uint64_t input_tensor_size;
  MemoryType output_tensor_loc;
  uint64_t output_tensor_size;
  uint64_t num_ops;
  CollectiveCommType comm_type;
  std::vector<bool> involved_dim;
  uint32_t communicator_id;
  uint32_t comm_src;
  uint32_t comm_dst;
  uint32_t comm_size;
  uint32_t comm_tag;
  uint32_t comm_priority;
};

} // namespace Chakra
