#pragma once

#include <cstddef>
#include <cstdint>

namespace alp {
  template<typename vec_t>
  class VectorStorage {
  public:
      using idx_t = int64_t;

      virtual idx_t add_vector(const vec_t *vec_ptr) = 0;

      virtual const vec_t *get_vector(idx_t id) const = 0;

      virtual size_t dimension() const = 0;

      virtual size_t size() const = 0;

      virtual ~VectorStorage() {}
  };
} // namespace alp
