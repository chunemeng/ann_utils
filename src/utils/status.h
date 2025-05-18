#pragma once

#include <memory>
#include <string_view>

namespace alp {

  class Status {
  public:
      enum Code : uint8_t {
          kOk = 0,
          kNotFound = 1,
          kCorruption = 2,
          kNotSupported = 3,
          kInvalidArgument = 4,
          kIOError = 5,
          kBGError = 6,
      };

  private:
      void AddMessage(std::string_view msg) {
          // 6 bytes are enough to store the length of the message and the code and the '\0' at the end
          state_ = msg;
      }

  public:
      Status() noexcept = default;

      Status(Code code) noexcept: code_(code) {
      }

      Status(Status &&s) noexcept: code_(s.code_), state_(std::move(s.state_)) {
          s.code_ = Code::kOk;
      }

      Status &operator=(Status &&s) noexcept {
          code_ = s.code_;
          state_ = std::move(s.state_);
          s.code_ = Code::kOk;
          return *this;
      }

      Status(const Status &s) noexcept = default;

      Status &operator=(const Status &s) noexcept = default;

      static Status OK() noexcept {
          return {Code::kOk};
      }

      static Status NotFound(std::string_view msg = {nullptr, 0}) noexcept {
          Status s;
          s.code_ = Code::kNotFound;
          if (!msg.empty()) {
              s.AddMessage(msg);
          }

          return s;
      }

      static Status IOError(std::string_view msg = {nullptr, 0}) noexcept {
          Status s;
          s.code_ = Code::kIOError;
          if (!msg.empty()) {
              s.AddMessage(msg);
          }

          return s;
      }

      static Status Corruption(std::string_view msg = {nullptr, 0}) noexcept {
          Status s;
          s.code_ = Code::kCorruption;
          if (!msg.empty()) {
              s.AddMessage(msg);
          }
          return s;
      }

      static Status BGError(std::string_view msg = {nullptr, 0}) noexcept {
          Status s;
          s.code_ = Code::kBGError;
          if (!msg.empty()) {
              s.AddMessage(msg);
          }
          return s;
      }

      static Status NotSupported() noexcept {
          return {Code::kNotSupported};
      }

      static Status InvalidArgument() noexcept {
          return {Code::kInvalidArgument};
      }

      [[nodiscard]] bool ok() const noexcept {
          return code_ == Code::kOk;
      }

      [[nodiscard]] bool IsNotFound() const noexcept {
          return code_ == Code::kNotFound;
      }

      [[nodiscard]] bool IsCorruption() const noexcept {
          return code_ == Code::kCorruption;
      }

      [[nodiscard]] bool IsNotSupport() const noexcept {
          return code_ == Code::kNotSupported;
      }

      [[nodiscard]] bool IsInvalidArgument() const noexcept {
          return code_ == Code::kInvalidArgument;
      }

      [[nodiscard]] bool IsIOError() const noexcept {
          return code_ == Code::kIOError;
      }

      [[nodiscard]] bool IsBGError() const noexcept {
          return code_ == Code::kBGError;
      }


      [[nodiscard]] Code code() const noexcept {
          return code_;
      }

      std::string_view ToString() const noexcept {
          return state_;
      }

  private:
      Code code_{};

      std::string state_{};
  };


}// namespace LSMKV