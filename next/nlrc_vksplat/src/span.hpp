#pragma once

#include <array>
#include <cstddef>
#include <type_traits>
#include <vector>

namespace nlrc::vksplat {

template <typename T>
class Span final {
public:
  using element_type = T; // NOLINT(readability-identifier-naming)

  constexpr Span() noexcept = default;
  constexpr Span(T *data, std::size_t size) noexcept : data_(data), size_(size) {}

  template <typename U, typename = std::enable_if_t<std::is_convertible_v<U *, T *>>>
  constexpr Span(Span<U> other) noexcept : data_(other.data()), size_(other.size()) {}

  [[nodiscard]] constexpr T *data() const noexcept {
    return data_;
  }

  [[nodiscard]] constexpr std::size_t size() const noexcept {
    return size_;
  }

  [[nodiscard]] constexpr std::size_t size_bytes() const noexcept {
    return size_ * sizeof(T);
  }

  [[nodiscard]] constexpr bool empty() const noexcept {
    return size_ == 0;
  }

  [[nodiscard]] constexpr T &operator[](std::size_t index) const noexcept {
    return data_[index];
  }

private:
  T *data_{};
  std::size_t size_{};
};

template <typename T>
[[nodiscard]] Span<T> make_span(T *data, std::size_t size) noexcept {
  return {data, size};
}

template <typename T>
[[nodiscard]] Span<T> make_span(std::vector<T> &values) noexcept {
  return {values.data(), values.size()};
}

template <typename T>
[[nodiscard]] Span<const T> make_span(const std::vector<T> &values) noexcept {
  return {values.data(), values.size()};
}

template <typename T, std::size_t Size>
[[nodiscard]] constexpr Span<T> make_span(std::array<T, Size> &values) noexcept {
  return {values.data(), values.size()};
}

template <typename T, std::size_t Size>
[[nodiscard]] constexpr Span<const T> make_span(const std::array<T, Size> &values) noexcept {
  return {values.data(), values.size()};
}

// NOLINTBEGIN(modernize-avoid-c-arrays)
template <typename T, std::size_t Size>
[[nodiscard]] constexpr Span<T> make_span(T (&values)[Size]) noexcept {
  return {values, Size};
}
// NOLINTEND(modernize-avoid-c-arrays)

// NOLINTBEGIN(modernize-avoid-c-arrays)
template <typename T, std::size_t Size>
[[nodiscard]] constexpr Span<const T> make_span(const T (&values)[Size]) noexcept {
  return {values, Size};
}
// NOLINTEND(modernize-avoid-c-arrays)

class ByteView final {
public:
  constexpr ByteView() noexcept = default;
  constexpr ByteView(const void *data, std::size_t size_bytes) noexcept : data_(data), size_bytes_(size_bytes) {}

  template <typename T>
  explicit constexpr ByteView(Span<T> values) noexcept : data_(values.data()), size_bytes_(values.size_bytes()) {
    static_assert(!std::is_void_v<T>, "ByteView element type must not be void");
  }

  template <typename T>
  [[nodiscard]] static constexpr ByteView from_object(const T &value) noexcept {
    return {&value, sizeof(T)};
  }

  [[nodiscard]] constexpr const void *data() const noexcept {
    return data_;
  }

  [[nodiscard]] constexpr std::size_t size_bytes() const noexcept {
    return size_bytes_;
  }

  [[nodiscard]] constexpr bool empty() const noexcept {
    return size_bytes_ == 0;
  }

private:
  const void *data_{};
  std::size_t size_bytes_{};
};

class MutableByteView final {
public:
  constexpr MutableByteView() noexcept = default;
  constexpr MutableByteView(void *data, std::size_t size_bytes) noexcept : data_(data), size_bytes_(size_bytes) {}

  template <typename T>
  explicit constexpr MutableByteView(Span<T> values) noexcept : data_(values.data()), size_bytes_(values.size_bytes()) {
    static_assert(!std::is_const_v<T>, "MutableByteView element type must be mutable");
  }

  template <typename T>
  [[nodiscard]] static constexpr MutableByteView from_object(T &value) noexcept {
    return {&value, sizeof(T)};
  }

  [[nodiscard]] constexpr void *data() const noexcept {
    return data_;
  }

  [[nodiscard]] constexpr std::size_t size_bytes() const noexcept {
    return size_bytes_;
  }

  [[nodiscard]] constexpr bool empty() const noexcept {
    return size_bytes_ == 0;
  }

private:
  void *data_{};
  std::size_t size_bytes_{};
};

} // namespace nlrc::vksplat
