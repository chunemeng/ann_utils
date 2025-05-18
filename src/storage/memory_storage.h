#pragma once

#include "storage.h"
#include <vector>
#include <cassert>

namespace alp {
  template<typename vec_t>
  class MemoryVectorStorage : public VectorStorage<vec_t> {
  public:
      using idx_t = typename VectorStorage<vec_t>::idx_t;

      explicit MemoryVectorStorage(size_t dim) : dim_(dim) {
          assert(dim > 0);
      }

      idx_t add_vector(const vec_t *vec_ptr) override {
          vectors_.insert(vectors_.end(), vec_ptr, vec_ptr + dim_);
          return size() - 1;
      }

      const vec_t *get_vector(idx_t id) const override {
          if (id < 0 || id >= size()) {
              return nullptr;
          }
          return &vectors_[id * dim_];
      }

      size_t dimension() const override {
          return dim_;
      }

      size_t size() const {
          return vectors_.size() / dim_;
      }

  private:
      size_t dim_;
      std::vector<vec_t> vectors_;
  };
} // namespace alp