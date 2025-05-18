#pragma once

#include <cstdint>
#include <vector>

#include "utils/status.h"

namespace alp {


  using idx_t = int64_t;

  template<typename vec_t>
  class VectorIndex {
  public:

      virtual ~VectorIndex() {}

      virtual Status add(idx_t id, const vec_t *vec_ptr) = 0;

      virtual Status add(const vec_t *vec_ptr) = 0;

      virtual Status build() = 0;

      virtual Status search(const vec_t *query_vec, size_t k,
                            std::vector<idx_t> &result_ids,
                            std::vector<float> &result_distances) const = 0;

      virtual size_t dimension() const = 0;

      virtual size_t size() const = 0;
  };
}

