/* Copyright 2025 SGLang Team. All Rights Reserved.
Licensed under the Apache License, Version 2.0. */

#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <torch/all.h>
#include <torch/library.h>

#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

int64_t dsv4_hisparse_cpu_attention(
    const at::Tensor& query,
    const at::Tensor& miss_locs,
    const at::Tensor& host_cache,
    double softmax_scale,
    int64_t head_dim_v,
    at::Tensor& output,
    at::Tensor& lse) {
  TORCH_CHECK(query.device().is_cpu() && query.scalar_type() == at::kBFloat16);
  TORCH_CHECK(miss_locs.device().is_cpu() && miss_locs.scalar_type() == at::kLong);
  TORCH_CHECK(host_cache.device().is_cpu() && host_cache.scalar_type() == at::kByte);
  TORCH_CHECK(query.is_contiguous() && miss_locs.is_contiguous());
  TORCH_CHECK(output.is_contiguous() && lse.is_contiguous());
  TORCH_CHECK(query.dim() == 3 && query.size(2) == 512);
  TORCH_CHECK(miss_locs.dim() == 2 && miss_locs.size(0) == query.size(0));
  TORCH_CHECK(head_dim_v > 0 && head_dim_v <= 512);
  TORCH_CHECK(
      output.dim() == 3 && output.size(0) == query.size(0) && output.size(1) == query.size(1) &&
      output.size(2) == head_dim_v);
  TORCH_CHECK(lse.dim() == 2 && lse.size(0) == query.size(0) && lse.size(1) == query.size(1));

  constexpr int64_t kPageSize = 64;
  constexpr int64_t kDataBytes = 576;
  constexpr int64_t kNopeDim = 448;
  constexpr int64_t kDim = 512;
  constexpr int64_t kScaleBytes = 8;
  const auto batches = query.size(0);
  const auto heads = query.size(1);
  const auto max_misses = miss_locs.size(1);
  const auto page_stride = host_cache.stride(0);
  const auto* q = query.const_data_ptr<at::BFloat16>();
  const auto* locs = miss_locs.const_data_ptr<int64_t>();
  const auto* cache = host_cache.const_data_ptr<uint8_t>();
  auto* out = output.mutable_data_ptr<at::BFloat16>();
  auto* out_lse = lse.mutable_data_ptr<float>();

  auto fp8 = [](uint8_t bits) -> float {
    const float sign = bits & 0x80 ? -1.0f : 1.0f;
    const int exp = (bits >> 3) & 0xf;
    const int mantissa = bits & 7;
    if (exp == 0xf && mantissa == 7) return std::numeric_limits<float>::quiet_NaN();
    const float value =
        exp == 0 ? std::ldexp(static_cast<float>(mantissa), -9) : std::ldexp(1.0f + mantissa / 8.0f, exp - 7);
    return sign * value;
  };
  auto kv_value = [&](int64_t loc, int64_t dim) -> float {
    const int64_t page = loc / kPageSize;
    const int64_t offset = loc % kPageSize;
    const auto* row = cache + page * page_stride + offset * kDataBytes;
    if (dim >= kNopeDim) {
      at::BFloat16 value;
      std::memcpy(&value, row + kNopeDim + 2 * (dim - kNopeDim), sizeof(value));
      return static_cast<float>(value);
    }
    const auto* scales = cache + page * page_stride + kPageSize * kDataBytes + offset * kScaleBytes;
    const uint8_t scale_exp = scales[dim / 64];
    const float scale = scale_exp == 0xff ? std::numeric_limits<float>::quiet_NaN()
                                          : std::ldexp(1.0f, static_cast<int>(scale_exp) - 127);
    return fp8(row[dim]) * scale;
  };

  output.zero_();
  lse.fill_(-std::numeric_limits<float>::infinity());
  at::parallel_for(0, batches * heads, 1, [&](int64_t begin, int64_t end) {
    for (int64_t work = begin; work < end; ++work) {
      const int64_t b = work / heads;
      const int64_t h = work % heads;
      int64_t count = 0;
      while (count < max_misses && locs[b * max_misses + count] >= 0)
        ++count;
      if (count == 0) continue;
      // Keep scratch rows aligned for oneDNN/AMX-friendly 16/32-token shapes.
      const int64_t padded = (count + 31) & ~int64_t(31);
      std::vector<float> scores(padded, -std::numeric_limits<float>::infinity());
      std::vector<at::BFloat16> kv(padded * kDim);
      for (int64_t m = 0; m < count; ++m) {
        const int64_t loc = locs[b * max_misses + m];
        for (int64_t d = 0; d < kDim; ++d)
          kv[m * kDim + d] = at::BFloat16(kv_value(loc, d));
      }
      const auto* q_row = q + (b * heads + h) * kDim;
      float maximum = -std::numeric_limits<float>::infinity();
      for (int64_t m = 0; m < count; ++m) {
        float score = 0.0f;
        for (int64_t d = 0; d < kDim; ++d)
          score += static_cast<float>(q_row[d]) * static_cast<float>(kv[m * kDim + d]);
        scores[m] = score * static_cast<float>(softmax_scale);
        maximum = std::max(maximum, scores[m]);
      }
      float sum = 0.0f;
      for (int64_t m = 0; m < count; ++m)
        sum += std::exp(scores[m] - maximum);
      out_lse[b * heads + h] = maximum + std::log(sum);
      auto* out_row = out + (b * heads + h) * head_dim_v;
      for (int64_t d = 0; d < head_dim_v; ++d) {
        float value = 0.0f;
        for (int64_t m = 0; m < count; ++m)
          value += std::exp(scores[m] - maximum) / sum * static_cast<float>(kv[m * kDim + d]);
        out_row[d] = at::BFloat16(value);
      }
    }
  });
  int64_t miss_tokens = 0;
  for (int64_t b = 0; b < batches; ++b)
    for (int64_t m = 0; m < max_misses && locs[b * max_misses + m] >= 0; ++m)
      ++miss_tokens;
  return miss_tokens;
}

TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {
  m.def(
      "dsv4_hisparse_cpu_attention(Tensor query, Tensor miss_locs, Tensor host_cache, float softmax_scale, int "
      "head_dim_v, Tensor(a!) output, Tensor(b!) lse) -> int");
  m.impl("dsv4_hisparse_cpu_attention", torch::kCPU, &dsv4_hisparse_cpu_attention);
}
