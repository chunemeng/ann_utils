#pragma once

#include "ivf/ivfflat_index.h"

namespace alp {
  template<typename vec_t>
  std::unique_ptr<VectorIndex<vec_t>> make_ivf_index(
          ivf::ClusterType index_type, const std::string &distance_type,
          size_t dim, size_t nlist, size_t nprobe) {
      return std::make_unique<ivf::IvfIndex<vec_t> >
              (index_type, nlist, nprobe, dim, distance_type);
  }

}