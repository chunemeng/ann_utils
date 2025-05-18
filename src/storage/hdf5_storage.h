#pragma once

#include <H5Cpp.h>
#include <vector>
#include <stdexcept>
#include <memory>

namespace alp {


  template<typename vec_t>
  class VectorStorageHDF5 : public VectorStorage<vec_t> {
  public:
      using idx_t = typename VectorStorage<vec_t>::idx_t;

      VectorStorageHDF5(const std::string &filename, size_t dim, bool read_only = false)
              : filename_(filename), dim_(dim), read_only_(read_only) {
          if (read_only_) {
              file_ = std::make_unique<H5::H5File>(filename_, H5F_ACC_RDONLY);
              dataset_ = file_->openDataSet("vectors");
              H5::DataSpace space = dataset_.getSpace();
              hsize_t dims[2];
              space.getSimpleExtentDims(dims, nullptr);
              if (dims[1] != dim_) {
                  throw std::runtime_error("Dimension mismatch.");
              }
              num_vectors_ = dims[0];
              // 读入所有数据到内存缓存（为了演示，实际可优化）
              data_.resize(num_vectors_ * dim_);
              dataset_.read(data_.data(), H5::PredType::NATIVE_DOUBLE);
          } else {
              file_ = std::make_unique<H5::H5File>(filename_, H5F_ACC_TRUNC);
          }
      }

      void add_vector(idx_t id, const vec_t *vec_ptr) override {
          if (read_only_) {
              throw std::runtime_error("Cannot add vector in read-only mode.");
          }
          if (dim_ == 0) throw std::runtime_error("Dimension not set.");

          if (id != (idx_t) num_vectors_) {
              throw std::runtime_error("Only support append in order with consecutive IDs starting at 0.");
          }
          data_.insert(data_.end(), vec_ptr, vec_ptr + dim_);
          ++num_vectors_;
      }

      void save() {
          if (read_only_) return;
          if (num_vectors_ == 0) return;

          hsize_t dims[2] = {num_vectors_, dim_};
          H5::DataSpace dataspace(2, dims);
          dataset_ = file_->createDataSet("vectors", H5::PredType::NATIVE_DOUBLE, dataspace);
          dataset_.write(data_.data(), H5::PredType::NATIVE_DOUBLE);
      }

      const vec_t *get_vector(idx_t id) const override {
          if (id < 0 || id >= (idx_t) num_vectors_) {
              return nullptr;
          }
          return &data_[id * dim_];
      }

      size_t dimension() const override {
          return dim_;
      }

      ~VectorStorageHDF5() {
          // 保存数据（非只读模式）
          if (!read_only_) {
              save();
          }
      }

  private:
      std::string filename_;
      size_t dim_ = 0;
      bool read_only_ = false;

      std::unique_ptr <H5::H5File> file_;
      H5::DataSet dataset_;

      std::vector <vec_t> data_;
      size_t num_vectors_ = 0;
  };

}
