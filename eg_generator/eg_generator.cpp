#include <iostream>
#include <string>

#include "cxxopts.hpp"
#include "eg_generator/DependencyGraph.h"
#include "third_party/utils/protoio.hh"

using namespace std;
using namespace Chakra;

int num_npus;
int num_dims;

string get_filename(string exp_name, int npu_id) {
  return exp_name + "." + to_string(npu_id) + ".eg";
}

Chakra::CollectiveCommType get_comm_type(
    ChakraNode::CollectiveCommType comm_type) {
  return ALL_REDUCE;
}

void one_comp_node() {
  DependencyGraph *dg;
  GraphNode *node;

  for (int npu_id = 0; npu_id < num_npus; npu_id++) {
    dg = new DependencyGraph(get_filename(string(__func__), npu_id));

    node = dg->addNode(ChakraNode::COMP_NODE);
    node->name = "COMP_NODE";
    node->simulated_run_time = 5;

    dg->flushEG();
    delete dg;
  }
}

void two_comp_nodes_independent() {
  DependencyGraph *dg;
  GraphNode *node;

  for (int npu_id = 0; npu_id < num_npus; npu_id++) {
    dg = new DependencyGraph(get_filename(string(__func__), npu_id));

    node = dg->addNode(ChakraNode::COMP_NODE);
    node->name = "COMP_NODE";
    node->simulated_run_time = 5;

    node = dg->addNode(ChakraNode::COMP_NODE);
    node->name = "COMP_NODE";
    node->simulated_run_time = 5;

    dg->flushEG();
    delete dg;
  }
}

void two_comp_nodes_dependent() {
  DependencyGraph *dg;
  GraphNode *node1, *node2;

  for (int npu_id = 0; npu_id < num_npus; npu_id++) {
    dg = new DependencyGraph(get_filename(string(__func__), npu_id));

    node1 = dg->addNode(ChakraNode::COMP_NODE);
    node1->name = "COMP_NODE";
    node1->simulated_run_time = 5;

    node2 = dg->addNode(ChakraNode::COMP_NODE);
    node2->name = "COMP_NODE";
    node2->simulated_run_time = 5;

    dg->assignDep(node1, node2);

    dg->flushEG();
    delete dg;
  }
}

void one_comm_node_all_reduce() {
  DependencyGraph *dg;
  GraphNode *node;

  for (int npu_id = 0; npu_id < num_npus; npu_id++) {
    dg = new DependencyGraph(get_filename(string(__func__), npu_id));

    node = dg->addNode(ChakraNode::COMM_COLL_NODE);
    node->name = "COMM_COLL_NODE";
    for (int i = 0; i < num_dims; i++) {
      node->involved_dim.push_back(true);
    }
    node->comm_type = get_comm_type(ChakraNode::ALL_REDUCE);
    node->comm_size = 65536;

    dg->flushEG();
    delete dg;
  }
}

void one_comm_node_all_to_all() {
  DependencyGraph *dg;
  GraphNode *node;

  for (int npu_id = 0; npu_id < num_npus; npu_id++) {
    dg = new DependencyGraph(get_filename(string(__func__), npu_id));

    node = dg->addNode(ChakraNode::COMM_COLL_NODE);
    node->name = "COMM_COLL_NODE";
    for (int i = 0; i < num_dims; i++) {
      node->involved_dim.push_back(true);
    }
    node->comm_type = get_comm_type(ChakraNode::ALL_TO_ALL);
    node->comm_size = 65536;

    dg->flushEG();
    delete dg;
  }
}

void one_comm_node_all_gather() {
  DependencyGraph *dg;
  GraphNode *node;

  for (int npu_id = 0; npu_id < num_npus; npu_id++) {
    dg = new DependencyGraph(get_filename(string(__func__), npu_id));

    node = dg->addNode(ChakraNode::COMM_COLL_NODE);
    node->name = "COMM_COLL_NODE";
    for (int i = 0; i < num_dims; i++) {
      node->involved_dim.push_back(true);
    }
    node->comm_type = get_comm_type(ChakraNode::ALL_GATHER);
    node->comm_size = 65536;

    dg->flushEG();
    delete dg;
  }
}

void one_comm_node_reduce_scatter() {
  DependencyGraph *dg;
  GraphNode *node;

  for (int npu_id = 0; npu_id < num_npus; npu_id++) {
    dg = new DependencyGraph(get_filename(string(__func__), npu_id));

    node = dg->addNode(ChakraNode::COMM_COLL_NODE);
    node->name = "COMM_COLL_NODE";
    for (int i = 0; i < num_dims; i++) {
      node->involved_dim.push_back(true);
    }
    node->comm_type = get_comm_type(ChakraNode::REDUCE_SCATTER);
    node->comm_size = 65536;

    dg->flushEG();
    delete dg;
  }
}

void comm_nodes_single_send_single_recv(uint32_t comm_size) {
  DependencyGraph *dg;
  GraphNode *node;

  for (int npu_id = 0; npu_id < num_npus; npu_id++) {
    dg = new DependencyGraph(
        get_filename(string(__func__) + "." + to_string(comm_size),
          npu_id));

    node = dg->addNode(ChakraNode::COMM_SEND_NODE);
    node->name = "COMM_SEND_NODE";
    node->comm_src = npu_id;
    if (npu_id != num_npus-1)
      node->comm_dst = npu_id + 1;
    else
      node->comm_dst = 0;
    node->comm_size = comm_size;
    node->comm_tag = 0;

    node = dg->addNode(ChakraNode::COMM_RECV_NODE);
    node->name = "COMM_RECV_NODE";
    if (npu_id != 0)
      node->comm_src = npu_id - 1;
    else
      node->comm_src = num_npus - 1;
    node->comm_dst = npu_id;
    node->comm_size = comm_size;
    node->comm_tag = 0;

    dg->flushEG();
    delete dg;
  }
}

void invalid_node_case_1() {
  DependencyGraph *dg;
  GraphNode *comp_node, *invalid_node;

  for (int npu_id = 0; npu_id < num_npus; npu_id++) {
    dg = new DependencyGraph(get_filename(string(__func__), npu_id));

    comp_node = dg->addNode(ChakraNode::COMP_NODE);
    comp_node->name = "COMP_NODE";
    comp_node->simulated_run_time = 5;

    invalid_node = dg->addNode(ChakraNode::INVALID_NODE);
    invalid_node->name = "INVALID_NODE";

    dg->assignDep(comp_node, invalid_node);

    dg->flushEG();
    delete dg;
  }
}

void invalid_node_case_2() {
  DependencyGraph *dg;
  GraphNode *comp_node, *invalid_node;

  for (int npu_id = 0; npu_id < num_npus; npu_id++) {
    dg = new DependencyGraph(get_filename(string(__func__), npu_id));

    comp_node = dg->addNode(ChakraNode::COMP_NODE);
    comp_node->name = "COMP_NODE";
    comp_node->simulated_run_time = 5;

    invalid_node = dg->addNode(ChakraNode::INVALID_NODE);
    invalid_node->name = "INVALID_NODE";

    dg->assignDep(comp_node, invalid_node);

    comp_node = dg->addNode(ChakraNode::COMP_NODE);
    comp_node->name = "COMP_NODE";
    comp_node->simulated_run_time = 5;

    dg->assignDep(invalid_node, comp_node);

    dg->flushEG();
    delete dg;
  }
}

void invalid_node_case_3() {
  DependencyGraph *dg;
  GraphNode *comp_node, *invalid_node;

  for (int npu_id = 0; npu_id < num_npus; npu_id++) {
    dg = new DependencyGraph(get_filename(string(__func__), npu_id));

    invalid_node = dg->addNode(ChakraNode::INVALID_NODE);
    invalid_node->name = "INVALID_NODE";

    comp_node = dg->addNode(ChakraNode::COMP_NODE);
    comp_node->name = "COMP_NODE";
    comp_node->simulated_run_time = 5;

    dg->assignDep(invalid_node, comp_node);

    dg->flushEG();
    delete dg;
  }
}

void invalid_node_case_4() {
  DependencyGraph *dg;
  GraphNode *comp_node, *invalid_node;

  for (int npu_id = 0; npu_id < num_npus; npu_id++) {
    dg = new DependencyGraph(get_filename(string(__func__), npu_id));

    invalid_node = dg->addNode(ChakraNode::INVALID_NODE);
    invalid_node->name = "INVALID_NODE";

    comp_node = dg->addNode(ChakraNode::COMP_NODE);
    comp_node->name = "COMP_NODE";
    comp_node->simulated_run_time = 5;

    dg->assignDep(invalid_node, comp_node);

    comp_node = dg->addNode(ChakraNode::COMP_NODE);
    comp_node->name = "COMP_NODE";
    comp_node->simulated_run_time = 5;

    dg->assignDep(invalid_node, comp_node);

    dg->flushEG();
    delete dg;
  }
}

void three_layer_data_parallel() {
  DependencyGraph *dg;
  GraphNode *node;

  for (int npu_id = 0; npu_id < num_npus; npu_id++) {
    dg = new DependencyGraph(get_filename(string(__func__), npu_id));

    GraphNode *fwd_0 = dg->addNode(ChakraNode::COMP_NODE);
    fwd_0->name = "COMP_NODE_FWD_0";
    fwd_0->simulated_run_time = 5;

    GraphNode *fwd_1 = dg->addNode(ChakraNode::COMP_NODE);
    fwd_1->name = "COMP_NODE_FWD_1";
    fwd_1->simulated_run_time = 5;

    GraphNode *fwd_2 = dg->addNode(ChakraNode::COMP_NODE);
    fwd_2->name = "COMP_NODE_FWD_2";
    fwd_2->simulated_run_time = 5;

    dg->assignDep(fwd_0, fwd_1);
    dg->assignDep(fwd_1, fwd_2);

    GraphNode *bwd_wg_2 = dg->addNode(ChakraNode::COMP_NODE);
    bwd_wg_2->name = "COMP_NODE_BWD_WG_2";
    bwd_wg_2->simulated_run_time = 5;

    GraphNode *bwd_ig_2 = dg->addNode(ChakraNode::COMP_NODE);
    bwd_ig_2->name = "COMP_NODE_BWD_IG_2";
    bwd_ig_2->simulated_run_time = 5;

    GraphNode *comm_2 = dg->addNode(ChakraNode::COMM_COLL_NODE);
    comm_2->name = "COMM_COLL_NODE_BWD_ALL_REDUCE_2";
    for (int i = 0; i < num_dims; i++) {
      comm_2->involved_dim.push_back(true);
    }
    comm_2->comm_type = get_comm_type(ChakraNode::ALL_REDUCE);
    comm_2->comm_size = 65536;

    GraphNode *bwd_wg_1 = dg->addNode(ChakraNode::COMP_NODE);
    bwd_wg_1->name = "COMP_NODE_BWD_WG_1";
    bwd_wg_1->simulated_run_time = 5;

    GraphNode *bwd_ig_1 = dg->addNode(ChakraNode::COMP_NODE);
    bwd_ig_1->name = "COMP_NODE_BWD_IG_1";
    bwd_ig_1->simulated_run_time = 5;

    GraphNode *comm_1 = dg->addNode(ChakraNode::COMM_COLL_NODE);
    comm_1->name = "COMM_COLL_NODE_BWD_ALL_REDUCE_1";
    for (int i = 0; i < num_dims; i++) {
      comm_1->involved_dim.push_back(true);
    }
    comm_1->comm_type = get_comm_type(ChakraNode::ALL_REDUCE);
    comm_1->comm_size = 65536;

    GraphNode *bwd_wg_0 = dg->addNode(ChakraNode::COMP_NODE);
    bwd_wg_0->name = "COMP_NODE_BWD_WG_0";
    bwd_wg_0->simulated_run_time = 5;

    GraphNode *comm_0 = dg->addNode(ChakraNode::COMM_COLL_NODE);
    comm_0->name = "COMM_COLL_NODE_BWD_ALL_REDUCE_0";
    for (int i = 0; i < num_dims; i++) {
      comm_0->involved_dim.push_back(true);
    }
    comm_0->comm_type = get_comm_type(ChakraNode::ALL_REDUCE);
    comm_0->comm_size = 65536;

    dg->assignDep(fwd_2, bwd_wg_2);
    dg->assignDep(bwd_wg_2, bwd_ig_2);
    dg->assignDep(bwd_ig_2, bwd_wg_1);
    dg->assignDep(bwd_wg_1, bwd_ig_1);
    dg->assignDep(bwd_ig_1, bwd_wg_0);

    dg->assignDep(bwd_wg_2, comm_2);
    dg->assignDep(bwd_wg_1, comm_1);
    dg->assignDep(bwd_wg_0, comm_0);

    dg->flushEG();
    delete dg;
  }
}

void three_layer_data_parallel_sequentially_dependent() {
  DependencyGraph *dg;
  GraphNode *node;

  for (int npu_id = 0; npu_id < num_npus; npu_id++) {
    dg = new DependencyGraph(get_filename(string(__func__), npu_id));

    int comm_src, comm_dst;
    if (npu_id != 0)
      comm_src = npu_id - 1;
    else
      comm_src = num_npus - 1;
    if (npu_id != num_npus-1)
      comm_dst = npu_id + 1;
    else
      comm_dst = 0;

    GraphNode *fwd_0 = dg->addNode(ChakraNode::COMP_NODE);
    fwd_0->name = "COMP_NODE_FWD_0";
    fwd_0->simulated_run_time = 5;

    GraphNode *fwd_1 = dg->addNode(ChakraNode::COMP_NODE);
    fwd_1->name = "COMP_NODE_FWD_1";
    fwd_1->simulated_run_time = 5;

    GraphNode *fwd_2 = dg->addNode(ChakraNode::COMP_NODE);
    fwd_2->name = "COMP_NODE_FWD_2";
    fwd_2->simulated_run_time = 5;

    dg->assignDep(fwd_0, fwd_1);
    dg->assignDep(fwd_1, fwd_2);

    GraphNode *bwd_wg_2 = dg->addNode(ChakraNode::COMP_NODE);
    bwd_wg_2->name = "COMP_NODE_BWD_WG_2";
    bwd_wg_2->simulated_run_time = 5;

    GraphNode *comm_2 = dg->addNode(ChakraNode::COMM_COLL_NODE);
    comm_2->name = "COMM_COLL_NODE_BWD_ALL_REDUCE_2";
    for (int i = 0; i < num_dims; i++) {
      comm_2->involved_dim.push_back(true);
    }
    comm_2->comm_type = get_comm_type(ChakraNode::ALL_REDUCE);
    comm_2->comm_size = 65536;

    GraphNode *bwd_ig_2 = dg->addNode(ChakraNode::COMP_NODE);
    bwd_ig_2->name = "COMP_NODE_BWD_IG_2";
    bwd_ig_2->simulated_run_time = 5;

    GraphNode *bwd_wg_1 = dg->addNode(ChakraNode::COMP_NODE);
    bwd_wg_1->name = "COMP_NODE_BWD_WG_1";
    bwd_wg_1->simulated_run_time = 5;

    GraphNode *comm_1 = dg->addNode(ChakraNode::COMM_COLL_NODE);
    comm_1->name = "COMM_COLL_NODE_BWD_ALL_REDUCE_1";
    for (int i = 0; i < num_dims; i++) {
      comm_1->involved_dim.push_back(true);
    }
    comm_1->comm_type = get_comm_type(ChakraNode::ALL_REDUCE);
    comm_1->comm_size = 65536;

    GraphNode *bwd_ig_1 = dg->addNode(ChakraNode::COMP_NODE);
    bwd_ig_1->name = "COMP_NODE_BWD_IG_1";
    bwd_ig_1->simulated_run_time = 5;

    GraphNode *bwd_wg_0 = dg->addNode(ChakraNode::COMP_NODE);
    bwd_wg_0->name = "COMP_NODE_BWD_WG_0";
    bwd_wg_0->simulated_run_time = 5;

    GraphNode *comm_0 = dg->addNode(ChakraNode::COMM_COLL_NODE);
    comm_0->name = "COMM_COLL_NODE_BWD_ALL_REDUCE_0";
    for (int i = 0; i < num_dims; i++) {
      comm_0->involved_dim.push_back(true);
    }
    comm_0->comm_type = get_comm_type(ChakraNode::ALL_REDUCE);
    comm_0->comm_size = 65536;

    dg->assignDep(fwd_2, bwd_wg_2);
    dg->assignDep(bwd_wg_2, comm_2);
    dg->assignDep(comm_2, bwd_ig_2);
    dg->assignDep(bwd_ig_2, bwd_wg_1);
    dg->assignDep(bwd_wg_1, comm_1);
    dg->assignDep(comm_1, bwd_ig_1);
    dg->assignDep(bwd_ig_1, bwd_wg_0);
    dg->assignDep(bwd_wg_0, comm_0);

    dg->flushEG();
    delete dg;
  }
}

void parallelism_comparison_data_parallel(
    uint64_t m, uint64_t k, uint64_t n, uint32_t data_type_size,
    uint64_t flops) {
  DependencyGraph *dg;

  int num_layers = num_npus;
  GraphNode *fwd_comp[num_layers],
  *bwd_wg_comp[num_layers], *bwd_ig_comp[num_layers],
  *bwd_wg_comm[num_layers];

  for (int npu_id = 0; npu_id < num_npus; npu_id++) {
    dg = new DependencyGraph(get_filename(string(__func__), npu_id));

    uint64_t fwd_simulated_run_time = (2 * (m / num_npus) * k * n) / flops;
    uint64_t bwd_simulated_run_time = fwd_simulated_run_time;
    uint32_t bwd_wg_comm_size = n * k * data_type_size;

    for (int i = 0; i < num_layers; i++) {
      fwd_comp[i] = dg->addNode(ChakraNode::COMP_NODE);
      fwd_comp[i]->name = "COMP_NODE_FWD_" + to_string(i);
      fwd_comp[i]->simulated_run_time = fwd_simulated_run_time;
    }

    for (int i = 0; i < num_layers; i++) {
      bwd_wg_comp[num_layers - i - 1] = dg->addNode(ChakraNode::COMP_NODE);
      bwd_wg_comp[num_layers - i - 1]->name = "COMP_NODE_BWD_WG_" + to_string(num_layers - i - 1);
      bwd_wg_comp[num_layers - i - 1]->simulated_run_time = bwd_simulated_run_time;

      bwd_wg_comm[num_layers - i - 1] = dg->addNode(ChakraNode::COMM_COLL_NODE);
      bwd_wg_comm[num_layers - i - 1]->name = "COMM_COLL_NODE_BWD_WG_ALL_REDUCE_" + to_string(num_layers - i - 1);
      for (int j = 0; j < num_dims; j++) {
        bwd_wg_comm[num_layers - i - 1]->involved_dim.push_back(true);
      }
      bwd_wg_comm[num_layers - i - 1]->comm_type = get_comm_type(ChakraNode::ALL_REDUCE);
      bwd_wg_comm[num_layers - i - 1]->comm_size = bwd_wg_comm_size;

      if (i != (num_layers - 1)) {
        bwd_ig_comp[num_layers - i - 1] = dg->addNode(ChakraNode::COMP_NODE);
        bwd_ig_comp[num_layers - i - 1]->name = "COMP_NODE_BWD_IG_" + to_string(num_layers - i - 1);
        bwd_ig_comp[num_layers - i - 1]->simulated_run_time = bwd_simulated_run_time;
      }
    }

    for (int i = 0; i < num_layers - 1; i++) {
      dg->assignDep(fwd_comp[i], fwd_comp[i+1]);
    }

    dg->assignDep(fwd_comp[num_layers-1], bwd_wg_comp[num_layers-1]);

    for (int i = 0; i < num_layers; i++) {
      if (i != 0) {
        dg->assignDep(bwd_wg_comp[i], bwd_ig_comp[i]);
      }
      dg->assignDep(bwd_wg_comp[i], bwd_wg_comm[i]);
    }

    for (int i = 1; i < num_layers; i++) {
      dg->assignDep(bwd_ig_comp[i], bwd_wg_comp[i-1]);
    }

    dg->flushEG();
    delete dg;
  }
}

void parallelism_comparison_model_parallel(
    uint64_t m, uint64_t k, uint64_t n, uint32_t data_type_size,
    uint64_t flops) {
  DependencyGraph *dg;

  int num_layers = num_npus;
  GraphNode *fwd_comp[num_layers], *fwd_comm[num_layers],
  *bwd_wg_comp[num_layers], *bwd_ig_comp[num_layers], *bwd_ig_comm[num_layers],
  *bwd_wg_comp_prev, *bwd_ig_comm_prev;

  for (int npu_id = 0; npu_id < num_npus; npu_id++) {
    dg = new DependencyGraph(get_filename(string(__func__), npu_id));

    uint64_t fwd_simulated_run_time = (2 * m * k * (n / num_npus)) / flops;
    uint64_t bwd_simulated_run_time = fwd_simulated_run_time;
    uint32_t fwd_comm_size = m * (n / num_npus) * data_type_size;
    uint32_t bwd_ig_comm_size = m * k * data_type_size;

    for (int i = 0; i < num_layers; i++) {
      fwd_comp[i] = dg->addNode(ChakraNode::COMP_NODE);
      fwd_comp[i]->name = "COMP_NODE_FWD_" + to_string(i);
      fwd_comp[i]->simulated_run_time = fwd_simulated_run_time;

      fwd_comm[i] = dg->addNode(ChakraNode::COMM_COLL_NODE);
      fwd_comm[i]->name = "COMM_COLL_NODE_FWD_ALL_GATHER_" + to_string(i);
      for (int j = 0; j < num_dims; j++) {
        fwd_comm[i]->involved_dim.push_back(true);
      }
      fwd_comm[i]->comm_type = get_comm_type(ChakraNode::ALL_GATHER);
      fwd_comm[i]->comm_size = fwd_comm_size;

      dg->assignDep(fwd_comp[i], fwd_comm[i]);
      if (i != 0) {
        dg->assignDep(fwd_comm[i-1], fwd_comp[i]);
      }
    }

    for (int i = 0; i < num_layers; i++) {
      bwd_ig_comp[num_layers - i - 1] = dg->addNode(ChakraNode::COMP_NODE);
      bwd_ig_comp[num_layers - i - 1]->name = "COMP_NODE_BWD_IG_" + to_string(num_layers - i - 1);
      bwd_ig_comp[num_layers - i - 1]->simulated_run_time = bwd_simulated_run_time;
      if (i == 0) {
        dg->assignDep(fwd_comm[num_layers-1], bwd_ig_comp[num_layers - i - 1]);
      }

      bwd_wg_comp[num_layers - i - 1] = dg->addNode(ChakraNode::COMP_NODE);
      bwd_wg_comp[num_layers - i - 1]->name = "COMP_NODE_BWD_WG_" + to_string(num_layers - i - 1);
      bwd_wg_comp[num_layers - i - 1]->simulated_run_time = bwd_simulated_run_time;

      bwd_ig_comm[num_layers - i - 1] = dg->addNode(ChakraNode::COMM_COLL_NODE);
      bwd_ig_comm[num_layers - i - 1]->name = "COMM_COLL_NODE_BWD_IG_" + to_string(num_layers - i - 1);
      for (int j = 0; j < num_dims; j++) {
        bwd_ig_comm[num_layers - i - 1]->involved_dim.push_back(true);
      }
      bwd_ig_comm[num_layers - i - 1]->comm_type = get_comm_type(ChakraNode::ALL_REDUCE);
      bwd_ig_comm[num_layers - i - 1]->comm_size = bwd_ig_comm_size;

      dg->assignDep(bwd_ig_comp[num_layers - i - 1], bwd_wg_comp[num_layers - i - 1]);
      dg->assignDep(bwd_ig_comp[num_layers - i - 1], bwd_ig_comm[num_layers - i - 1]);

      if (i != 0) {
        dg->assignDep(bwd_wg_comp_prev, bwd_ig_comp[num_layers - i - 1]);
        dg->assignDep(bwd_ig_comm_prev, bwd_ig_comp[num_layers - i - 1]);
      }

      bwd_wg_comp_prev = bwd_wg_comp[num_layers - i - 1];
      bwd_ig_comm_prev = bwd_ig_comm[num_layers - i - 1];
    }

    dg->flushEG();
    delete dg;
  }
}

uint32_t parallelism_comparison_pipeline_parallel_get_tag(
    uint32_t src_npu_id, uint32_t dst_npu_id, uint32_t minibatch_id, uint32_t is_fwd)
{
  uint32_t tag = (
      ((src_npu_id & 0x3ff) << 22)
      | ((dst_npu_id & 0x3ff) << 12)
      | ((minibatch_id & 0x3ff) << 2)
      | (is_fwd & 0x3));
  return tag;
}

// GPipe
void parallelism_comparison_pipeline_parallel(
    uint64_t m, uint64_t k, uint64_t n, uint32_t data_type_size,
    uint64_t flops,
    uint32_t num_microbatches) {
  DependencyGraph *dg;
  GraphNode *fwd_comp, *fwd_comp_prev, *fwd_send, *fwd_recv, *fwd_recv_prev,
            *bwd_wg_comp, *bwd_wg_comp_prev, *bwd_ig_comp, *bwd_ig_comp_prev,
            *bwd_ig_send, *bwd_ig_recv, *bwd_ig_recv_prev;

  for (int npu_id = 0; npu_id < num_npus; npu_id++) {
    dg = new DependencyGraph(get_filename(string(__func__), npu_id));

    uint64_t fwd_simulated_run_time =
      (2 * (m / num_microbatches) * k * n) / flops;
    uint64_t bwd_simulated_run_time = fwd_simulated_run_time;
    uint32_t fwd_comm_size =
      (m / num_microbatches) * n * data_type_size;
    uint32_t bwd_ig_comm_size =
      (m / num_microbatches) * k * data_type_size;

    if (npu_id == 0) {
      for (uint32_t mb = 0; mb < num_microbatches; mb++) {
        fwd_comp = dg->addNode(ChakraNode::COMP_NODE);
        fwd_comp->name = "COMP_NODE_MB_" + to_string(mb) + "_FWD";
        fwd_comp->simulated_run_time = fwd_simulated_run_time;
        if (mb != 0) {
          dg->assignDep(fwd_comp_prev, fwd_comp);
        }
        fwd_comp_prev = fwd_comp;

        fwd_send = dg->addNode(ChakraNode::COMM_SEND_NODE);
        fwd_send->name = "COMM_SEND_NODE_MB_" + to_string(mb) + "_FWD";
        fwd_send->comm_src = 0;
        fwd_send->comm_dst = 1;
        fwd_send->comm_size = fwd_comm_size;
        fwd_send->comm_tag = parallelism_comparison_pipeline_parallel_get_tag(0, 1, mb, 1);
        dg->assignDep(fwd_comp, fwd_send);
      }

      for (uint32_t mb = 0; mb < num_microbatches; mb++) {
        bwd_ig_recv = dg->addNode(ChakraNode::COMM_RECV_NODE);
        bwd_ig_recv->name = "COMM_RECV_NODE_MB_" + to_string(mb) + "_BWD";
        bwd_ig_recv->comm_src = 1;
        bwd_ig_recv->comm_dst = 0;
        bwd_ig_recv->comm_size = bwd_ig_comm_size;
        bwd_ig_recv->comm_tag = parallelism_comparison_pipeline_parallel_get_tag(1, 0, mb, 0);
        if (mb == 0) {
          dg->assignDep(fwd_send, bwd_ig_recv);
        } else {
          dg->assignDep(bwd_ig_recv_prev, bwd_ig_recv);
        }
        bwd_ig_recv_prev = bwd_ig_recv;

        bwd_wg_comp = dg->addNode(ChakraNode::COMP_NODE);
        bwd_wg_comp->name = "COMP_NODE_MB_" + to_string(mb) + "_BWD_WG";
        bwd_wg_comp->simulated_run_time = bwd_simulated_run_time;
        dg->assignDep(bwd_ig_recv, bwd_wg_comp);
      }
    } else if (npu_id == (num_npus - 1)) {
      for (uint32_t mb = 0; mb < num_microbatches; mb++) {
        fwd_recv = dg->addNode(ChakraNode::COMM_RECV_NODE);
        fwd_recv->name = "COMM_RECV_NODE_MB_" + to_string(mb) + "_FWD";
        fwd_recv->comm_src = num_npus - 2;
        fwd_recv->comm_dst = num_npus - 1;
        fwd_recv->comm_size = fwd_comm_size;
        fwd_recv->comm_tag = parallelism_comparison_pipeline_parallel_get_tag(
            num_npus - 2, num_npus - 1, mb, 1);
        if (mb != 0) {
          dg->assignDep(fwd_recv_prev, fwd_recv);
        }
        fwd_recv_prev = fwd_recv;

        fwd_comp = dg->addNode(ChakraNode::COMP_NODE);
        fwd_comp->name = "COMP_NODE_MB_" + to_string(mb) + "_FWD_COMP";
        fwd_comp->simulated_run_time = fwd_simulated_run_time;
        fwd_comp_prev = fwd_comp;
        dg->assignDep(fwd_recv, fwd_comp);
      }

      for (uint32_t mb = 0; mb < num_microbatches; mb++) {
        bwd_ig_comp = dg->addNode(ChakraNode::COMP_NODE);
        bwd_ig_comp->name = "COMP_NODE_MB_" + to_string(mb) + "_BWD_IG";
        bwd_ig_comp->simulated_run_time = bwd_simulated_run_time;
        if (mb == 0) {
          dg->assignDep(fwd_comp, bwd_ig_comp);
        } else {
          dg->assignDep(bwd_wg_comp_prev, bwd_ig_comp);
        }

        bwd_wg_comp = dg->addNode(ChakraNode::COMP_NODE);
        bwd_wg_comp->name = "COMP_NODE_MB_" + to_string(mb) + "_BWD_WG";
        bwd_wg_comp->simulated_run_time = bwd_simulated_run_time;
        dg->assignDep(bwd_ig_comp, bwd_wg_comp);
        bwd_wg_comp_prev = bwd_wg_comp;

        bwd_ig_send = dg->addNode(ChakraNode::COMM_SEND_NODE);
        bwd_ig_send->name = "COMM_SEND_NODE_MB_" + to_string(mb) + "_BWD_IG";
        bwd_ig_send->comm_src = num_npus - 1;
        bwd_ig_send->comm_dst = num_npus - 2;
        bwd_ig_send->comm_size = bwd_ig_comm_size;
        bwd_ig_send->comm_tag = parallelism_comparison_pipeline_parallel_get_tag(
            num_npus - 1, num_npus - 2, mb, 0);
        dg->assignDep(bwd_ig_comp, bwd_ig_send);
      }
    } else {
      for (uint32_t mb = 0; mb < num_microbatches; mb++) {
        fwd_recv = dg->addNode(ChakraNode::COMM_RECV_NODE);
        fwd_recv->name = "COMM_RECV_NODE_MB_" + to_string(mb) + "_FWD";
        fwd_recv->comm_src = npu_id - 1;
        fwd_recv->comm_dst = npu_id;
        fwd_recv->comm_size = fwd_comm_size;
        fwd_recv->comm_tag = parallelism_comparison_pipeline_parallel_get_tag(
            npu_id - 1, npu_id, mb, 1);
        if (mb != 0) {
          dg->assignDep(fwd_recv_prev, fwd_recv);
        }
        fwd_recv_prev = fwd_recv;

        fwd_comp = dg->addNode(ChakraNode::COMP_NODE);
        fwd_comp->name = "COMP_NODE_MB_" + to_string(mb) + "_FWD";
        fwd_comp->simulated_run_time = fwd_simulated_run_time;
        fwd_comp_prev = fwd_comp;
        dg->assignDep(fwd_recv, fwd_comp);

        fwd_send = dg->addNode(ChakraNode::COMM_SEND_NODE);
        fwd_send->name = "COMM_SEND_NODE_MB_" + to_string(mb) + "_FWD";
        fwd_send->comm_src = npu_id;
        fwd_send->comm_dst = npu_id + 1;
        fwd_send->comm_size = fwd_comm_size;
        fwd_send->comm_tag = parallelism_comparison_pipeline_parallel_get_tag(
            npu_id, npu_id + 1, mb, 1);
        dg->assignDep(fwd_comp, fwd_send);
      }

      for (uint32_t mb = 0; mb < num_microbatches; mb++) {
        bwd_ig_recv = dg->addNode(ChakraNode::COMM_RECV_NODE);
        bwd_ig_recv->name = "COMM_RECV_NODE_MB_" + to_string(mb) + "_BWD_IG";
        bwd_ig_recv->comm_src = npu_id + 1;
        bwd_ig_recv->comm_dst = npu_id;
        bwd_ig_recv->comm_size = bwd_ig_comm_size;
        bwd_ig_recv->comm_tag = parallelism_comparison_pipeline_parallel_get_tag(
            npu_id + 1, npu_id, mb, 0);
        if (mb == 0) {
          dg->assignDep(fwd_send, bwd_ig_recv);
        } else {
          dg->assignDep(bwd_ig_recv_prev, bwd_ig_recv);
        }
        bwd_ig_recv_prev = bwd_ig_recv;

        bwd_ig_comp = dg->addNode(ChakraNode::COMP_NODE);
        bwd_ig_comp->name = "COMP_NODE_MB_" + to_string(mb) + "_BWD_IG";
        bwd_ig_comp->simulated_run_time = bwd_simulated_run_time;
        dg->assignDep(bwd_ig_recv, bwd_ig_comp);

        bwd_wg_comp = dg->addNode(ChakraNode::COMP_NODE);
        bwd_wg_comp->name = "COMP_NODE_MB_" + to_string(mb) + "_BWD_WG";
        bwd_wg_comp->simulated_run_time = bwd_simulated_run_time;
        dg->assignDep(bwd_ig_comp, bwd_wg_comp);
        bwd_wg_comp_prev = bwd_wg_comp;

        bwd_ig_send = dg->addNode(ChakraNode::COMM_SEND_NODE);
        bwd_ig_send->name = "COMM_SEND_NODE_MB_" + to_string(mb) + "_BWD_IG";
        bwd_ig_send->comm_src = npu_id;
        bwd_ig_send->comm_dst = npu_id - 1;
        bwd_ig_send->comm_size = bwd_ig_comm_size;
        bwd_ig_send->comm_tag = parallelism_comparison_pipeline_parallel_get_tag(
            npu_id, npu_id - 1, mb, 0);
        dg->assignDep(bwd_ig_comp, bwd_ig_send);
      }
    }

    dg->flushEG();
    delete dg;
  }
}

int main(int argc, char **argv)
{
  cxxopts::Options options("graphgen", "generates example execution graphs");

  options.add_options()
    ("num_npus", "Number of NPUs",
     cxxopts::value<int>()->default_value("64"))
    ("num_dims", "Number of dimensions in the network topology",
     cxxopts::value<int>()->default_value("2"))
    ;
  auto result = options.parse(argc, argv);
  num_npus = result["num_npus"].as<int>();
  num_dims = result["num_dims"].as<int>();

  one_comp_node();
  two_comp_nodes_independent();
  two_comp_nodes_dependent();

  one_comm_node_all_reduce();
  one_comm_node_all_to_all();
  one_comm_node_all_gather();
  one_comm_node_reduce_scatter();

  for (uint32_t i = 6; i < 17; i++) {
    comm_nodes_single_send_single_recv(1 << i);
  }

  invalid_node_case_1();
  invalid_node_case_2();
  invalid_node_case_3();
  invalid_node_case_4();

  three_layer_data_parallel();
  three_layer_data_parallel_sequentially_dependent();

  // parallelism comparison
  uint64_t flops = 2*512*512;
  parallelism_comparison_data_parallel(512, 512, 512, 2, flops);
  parallelism_comparison_model_parallel(512, 512, 512, 2, flops);
  parallelism_comparison_pipeline_parallel(512, 512, 512, 2, flops, 64);

  return 0;
}
