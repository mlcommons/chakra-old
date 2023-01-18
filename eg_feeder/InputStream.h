#pragma once

#include <string>

#include "eg_format/GraphNode.h"
#include "third_party/utils/protoio.hh"

namespace Chakra {

class InputStream {
 public:
  InputStream(const std::string& filename);
  void reset();
  bool read(GraphNode* element);

 private:
  ProtoInputStream trace;
};

} // namespace Chakra
