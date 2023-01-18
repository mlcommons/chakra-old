#include "eg_feeder/GraphFeeder.h"

using namespace std;
using namespace Chakra;

GraphFeeder::GraphFeeder(string filename) {
  is = new InputStream(filename);
  window_size = GraphFeederWindowSize;
  eg_complete = false;
  readNextWindow();
}

GraphFeeder::~GraphFeeder() {
  delete is;
}

void GraphFeeder::addNode(GraphNode* node) {
  dep_graph[node->id] = (GraphNode*)node;
}

void GraphFeeder::removeNode(NodeId node_id) {
  dep_graph.erase(node_id);

  if (!eg_complete
      && (dep_free_queue.size() < window_size)) {
    readNextWindow();
  }
}

bool GraphFeeder::hasNodesToIssue() {
  return !(dep_graph.empty() && dep_free_queue.empty());
}

GraphNode* GraphFeeder::getNextIssuableNode() {
  if (dep_free_queue.size() != 0) {
    GraphNode* node = dep_free_queue.front();
    dep_free_queue.pop();
    return node;
  } else {
    return nullptr;
  }
}

void GraphFeeder::pushBackIssuableNode(NodeId node_id) {
  GraphNode* node = dep_graph[node_id];
  dep_free_queue.push(node);
}

GraphNode* GraphFeeder::lookupNode(NodeId node_id) {
  return dep_graph[node_id];
}

void GraphFeeder::freeChildrenNodes(NodeId node_id) {
  GraphNode* node = dep_graph[node_id];
  for (auto child: node->dependents) {
    if (child->removeDepOn(node->id)) {
      dep_free_queue.push((GraphNode*)child);
    }
  }
}

void GraphFeeder::readNextWindow() {
  uint32_t num_read = 0;
  while (num_read != window_size) {
    GraphNode* new_node = new GraphNode;

    if (!is->read(new_node)) {
      eg_complete = true;
      delete new_node;
      return;
    }

    addNode(new_node);
    addDepsOnParent(new_node, new_node->dep_list);
    if (new_node->dep_list.empty()) {
      dep_free_queue.push(new_node);
    }

    num_read++;
  }
}

template<typename T>
void GraphFeeder::addDepsOnParent(GraphNode* new_node, T& dep_list) {
  auto dep_it = dep_list.begin();
  while (dep_it != dep_list.end()) {
    auto parent_itr = dep_graph.find(*dep_it);
    if (parent_itr != dep_graph.end()) {
      parent_itr->second->dependents.push_back(new_node);
      auto num_depts = parent_itr->second->dependents.size();
      dep_it++;
    } else {
      dep_it = dep_list.erase(dep_it);
    }
  }
}
