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

#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "eg_format/GraphNode.h"
#include "eg_format/execution_graph_def.pb.h"

typedef ProtoMessage::ChakraNode ChakraNode;

namespace Chakra {

class DependencyGraph {
 public:
  DependencyGraph(const std::string filename);
  GraphNode* addNode(ChakraNode::NodeType node_type);
  void assignDep(uint64_t past_node_id, uint64_t new_node_id);
  void assignDep(GraphNode* past_node, GraphNode* new_node);
  void writeTrace(uint32_t num_to_write);
  void flushEG();

 private:
  typedef typename std::vector<GraphNode*>::iterator depTraceItr;

  Chakra::GraphNodeType convertProtobufNodeType2CppNodeType(
          ChakraNode::NodeType node_type);
  ChakraNode::NodeType convertCppNodeType2ProtobufNodeType(
          Chakra::GraphNodeType node_type);
  Chakra::MemoryType convertProtobufMemType2CppMemType(
          ChakraNode::MemoryType memory_type);
  ChakraNode::MemoryType convertCppMemType2ProtobufMemType(
          Chakra::MemoryType memory_type);
  Chakra::CollectiveCommType convertProtobufCommType2CppCommType(
          ChakraNode::CollectiveCommType comm_type);
  ChakraNode::CollectiveCommType convertCppCommType2ProtobufCommType(
          Chakra::CollectiveCommType comm_type);

  std::vector<GraphNode*> depTrace;
  std::unordered_map<uint64_t, GraphNode*> graphInfoMap;
  ProtoOutputStream* graphStream;
  bool firstWin;
  uint32_t depWindowSize;
  uint64_t currID;
};

} // namespace Chakra
