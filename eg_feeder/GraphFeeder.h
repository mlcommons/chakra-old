#pragma once

#include <queue>
#include <unordered_map>

#include "eg_format/GraphNode.h"
#include "eg_feeder/InputStream.h"

namespace Chakra {

constexpr uint32_t GraphFeederWindowSize = 4096;

class GraphFeeder {
 public:
  GraphFeeder(std::string filename);
  ~GraphFeeder();

  void addNode(GraphNode* node);
  void removeNode(NodeId node_id);
  bool hasNodesToIssue();
  GraphNode* getNextIssuableNode();
  void pushBackIssuableNode(NodeId node_id);
  GraphNode* lookupNode(NodeId node_id);
  void freeChildrenNodes(NodeId node_id);

 private:
  void readNextWindow();
  template<typename T>
  void addDepsOnParent(GraphNode* new_node, T& dep_list);

  InputStream* is;
  uint32_t window_size;
  bool eg_complete;

  std::unordered_map<NodeId, GraphNode*> dep_graph;
  std::queue<GraphNode*> dep_free_queue;
};

} // namespace Chakra
