/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "error_check_macros.h"
#include "hip_custom_kernels.h"
#include "runtime_types.h"
#include "zp_unpack_cache.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define HIP_CHECK(cmd) HIP_CHECK_GOTO(cmd, cleanup)

namespace {

struct QMoEExpertCacheEntry {
  int64_t indices[4] = {};
  void *device_base = nullptr;
  size_t total_size = 0;
  size_t capacity = 0;
  size_t table_offsets[4] = {};
  std::vector<int32_t> expert_for_slot;
  std::vector<size_t> last_used;
  std::vector<size_t> use_count;
  size_t tick = 0;
};

struct QMoEExpertCache {
  size_t budget_bytes = 0;
  size_t allocated_bytes = 0;
  bool allocation_failed = false;
  std::vector<QMoEExpertCacheEntry> entries;
};

static size_t qmoe_expert_cache_budget() {
  const char *value = std::getenv("HIPDNN_EP_QMOE_EXPERT_CACHE_MIB");
  if (!value || !*value)
    return 0;
  char *end = nullptr;
  const unsigned long long mib = std::strtoull(value, &end, 10);
  if (end == value || *end != '\0' || mib > SIZE_MAX / (1024ull * 1024ull))
    return 0;
  return static_cast<size_t>(mib) * 1024ull * 1024ull;
}

static size_t qmoe_expert_cache_slots() {
  const char *value = std::getenv("HIPDNN_EP_QMOE_EXPERT_CACHE_SLOTS");
  if (!value || !*value)
    return 24;
  char *end = nullptr;
  const unsigned long long slots = std::strtoull(value, &end, 10);
  if (end == value || *end != '\0' || slots == 0 || slots > SIZE_MAX)
    return 24;
  return static_cast<size_t>(slots);
}

static QMoEExpertCache *get_qmoe_expert_cache(RuntimeState *state) {
  auto *cache = static_cast<QMoEExpertCache *>(
      hipdnn_ep_state_get_qmoe_expert_cache(state));
  if (cache)
    return cache;
  const size_t budget = qmoe_expert_cache_budget();
  if (budget == 0)
    return nullptr;
  cache = new QMoEExpertCache();
  cache->budget_bytes = budget;
  hipdnn_ep_state_set_qmoe_expert_cache(state, cache);
  return cache;
}

static QMoEExpertCacheEntry *find_or_create_cache_entry(
    RuntimeState *state, const int64_t indices[4], const size_t expert_sizes[4],
    int64_t num_experts, int64_t minimum_slots) {
  QMoEExpertCache *cache = get_qmoe_expert_cache(state);
  if (!cache)
    return nullptr;
  for (auto &entry : cache->entries) {
    if (entry.indices[0] == indices[0] && entry.indices[1] == indices[1] &&
        entry.indices[2] == indices[2] && entry.indices[3] == indices[3])
      return &entry;
  }
  if (cache->allocation_failed || minimum_slots <= 0)
    return nullptr;

  constexpr size_t kTableAlignment = 256;
  const size_t min_capacity = static_cast<size_t>(minimum_slots);
  size_t capacity = std::min(static_cast<size_t>(num_experts),
                             std::max(min_capacity, qmoe_expert_cache_slots()));
  size_t table_offsets[4] = {};
  size_t total_size = 0;
  auto compute_layout = [&](size_t slots) {
    total_size = 0;
    for (int i = 0; i < 4; ++i) {
      total_size = (total_size + kTableAlignment - 1) &
                   ~(kTableAlignment - 1);
      table_offsets[i] = total_size;
      if (expert_sizes[i] > SIZE_MAX / slots)
        return false;
      const size_t bytes = expert_sizes[i] * slots;
      if (bytes > SIZE_MAX - total_size)
        return false;
      total_size += bytes;
    }
    return true;
  };
  while (capacity >= min_capacity) {
    if (compute_layout(capacity) && total_size <= cache->budget_bytes &&
        cache->allocated_bytes <= cache->budget_bytes - total_size)
      break;
    --capacity;
  }
  if (capacity < min_capacity)
    return nullptr;

  void *device_base = nullptr;
  if (hipMalloc(&device_base, total_size) != hipSuccess) {
    cache->allocation_failed = true;
    RUNTIME_DEBUG_LOG(
        "[REAL] QMoE expert cache allocation stopped at %zu bytes\n",
        cache->allocated_bytes);
    return nullptr;
  }
  QMoEExpertCacheEntry entry;
  for (int i = 0; i < 4; ++i)
    entry.indices[i] = indices[i];
  entry.device_base = device_base;
  entry.total_size = total_size;
  entry.capacity = capacity;
  for (int i = 0; i < 4; ++i)
    entry.table_offsets[i] = table_offsets[i];
  entry.expert_for_slot.resize(capacity, -1);
  entry.last_used.resize(capacity, 0);
  entry.use_count.resize(capacity, 0);
  cache->allocated_bytes += total_size;
  cache->entries.push_back(std::move(entry));
  RUNTIME_DEBUG_LOG(
      "[REAL] QMoE expert cache allocated layer %zu: %zu slots, %zu bytes "
      "(%zu/%zu total)\n",
      cache->entries.size(), capacity, total_size, cache->allocated_bytes,
      cache->budget_bytes);
  return &cache->entries.back();
}

} // namespace

void hipdnn_ep_qmoe_expert_cache_destroy(void *opaque_cache) {
  auto *cache = static_cast<QMoEExpertCache *>(opaque_cache);
  if (!cache)
    return;
  for (auto &entry : cache->entries) {
    if (entry.device_base)
      hipFree(entry.device_base);
  }
  delete cache;
}

int wrap_qmoe_from_file(
    RuntimeState *state, const void *input, const void *router_probs,
    const void *router_weights, int64_t fc1_weights_index,
    int64_t fc1_scales_index, const void *fc1_bias,
    int64_t fc2_weights_index, int64_t fc2_scales_index,
    const void *fc2_bias, const void *fc3_weights, const void *fc3_scales,
    const void *fc3_bias, const void *fc1_zero_points,
    const void *fc2_zero_points, const void *fc3_zero_points, void *output,
    int64_t num_tokens, int64_t hidden_size, int64_t inter_size,
    int64_t num_experts, int64_t k, int64_t expert_weight_bits,
    int64_t block_size, int64_t swiglu_fusion, int64_t activation_type,
    float activation_alpha, float activation_beta, float swiglu_limit,
    int64_t normalize_routing_weights, int64_t elem_size) {
  if (!state) {
    fprintf(stderr, "wrap_qmoe_from_file: null state\n");
    return -1;
  }

  const int64_t indices[4] = {fc1_weights_index, fc1_scales_index,
                              fc2_weights_index, fc2_scales_index};
  const void *host_tables[4] = {};
  size_t table_sizes[4] = {};
  size_t table_offsets[4] = {};
  size_t total_size = 0;
  constexpr size_t kTableAlignment = 256;
  for (int i = 0; i < 4; ++i) {
    host_tables[i] =
        hipdnn_ep_disk_constant_get(state, indices[i], &table_sizes[i]);
    if (!host_tables[i] || table_sizes[i] == 0) {
      fprintf(stderr,
              "wrap_qmoe_from_file: failed to access disk constant %lld\n",
              (long long)indices[i]);
      return -1;
    }
    if (total_size > SIZE_MAX - (kTableAlignment - 1)) {
      fprintf(stderr, "wrap_qmoe_from_file: table layout overflow\n");
      return -1;
    }
    total_size = (total_size + kTableAlignment - 1) &
                 ~(kTableAlignment - 1);
    table_offsets[i] = total_size;
    if (table_sizes[i] > SIZE_MAX - total_size) {
      fprintf(stderr, "wrap_qmoe_from_file: table size overflow\n");
      return -1;
    }
    total_size += table_sizes[i];
  }

  auto stream =
      static_cast<hipStream_t>(hipdnn_ep_state_get_stream(state));
  if (!stream) {
    fprintf(stderr, "wrap_qmoe_from_file: missing stream\n");
    return -1;
  }

  size_t expert_sizes[4] = {};
  size_t bytes_per_active_expert = 0;
  for (int i = 0; i < 4; ++i) {
    if (table_sizes[i] % static_cast<size_t>(num_experts) != 0) {
      fprintf(stderr,
              "wrap_qmoe_from_file: constant %lld size %zu is not divisible "
              "by %lld experts\n",
              (long long)indices[i], table_sizes[i], (long long)num_experts);
      return -1;
    }
    expert_sizes[i] = table_sizes[i] / static_cast<size_t>(num_experts);
    if (expert_sizes[i] > SIZE_MAX - bytes_per_active_expert) {
      fprintf(stderr, "wrap_qmoe_from_file: expert staging size overflow\n");
      return -1;
    }
    bytes_per_active_expert += expert_sizes[i];
  }

  // The persistent cache is decode-only. Prefill touches many experts and is
  // better served by the transient sparse streaming path.
  QMoEExpertCacheEntry *cache_entry =
      num_tokens == 1
          ? find_or_create_cache_entry(state, indices, expert_sizes,
                                       num_experts, k)
          : nullptr;

  // Route first so file-backed tables only upload expert slices used by this
  // invocation. The regular wrapper repeats this inexpensive deterministic
  // routing launch after the copies, keeping its public ABI unchanged.
  const size_t routed_count = static_cast<size_t>(num_tokens * k);
  const size_t route_indices_bytes = routed_count * sizeof(int32_t);
  const size_t route_weights_offset =
      (route_indices_bytes + 63) & ~size_t(63);
  const size_t route_scratch_bytes = route_weights_offset + routed_count * elem_size;
  // Decode later reuses these routing buffers inside wrap_qmoe. Reserve a
  // conservative floor before routing so wrap_qmoe's full scratch layout does
  // not grow the allocation and invalidate the pointers. Typical decode uses
  // well under 1 MiB; 16 MiB also covers much larger hidden/inter sizes.
  const size_t route_allocation_bytes =
      num_tokens == 1 ? std::max(route_scratch_bytes, size_t(16) << 20)
                      : route_scratch_bytes;
  if (hipdnn_ep_state_ensure_qmoe_scratch(state, route_allocation_bytes) != 0) {
    fprintf(stderr, "wrap_qmoe_from_file: unable to allocate routing scratch\n");
    return -1;
  }
  auto *route_scratch =
      static_cast<uint8_t *>(hipdnn_ep_state_get_qmoe_scratch(state));
  void *route_indices = route_scratch;
  void *route_weights = route_scratch + route_weights_offset;
  if (hip_qmoe_topk_routing(stream, router_probs, route_indices, route_weights,
                            num_tokens, num_experts, k,
                            normalize_routing_weights, elem_size) != 0) {
    fprintf(stderr, "wrap_qmoe_from_file: top-k routing failed\n");
    return -1;
  }

  // Decode copies expert slices through pinned staging. Pageable file mappings
  // make hipMemcpyAsync stage every small expert copy internally; explicitly
  // packing the 8 active experts into one reusable pinned allocation makes the
  // H2D copies genuinely asynchronous. The routing D2H sync at the start of
  // the next layer guarantees this staging is no longer in flight before the
  // CPU reuses it.
  const bool use_pinned_staging = num_tokens == 1;
  uint8_t *host_scratch = nullptr;
  size_t packed_offset = (route_indices_bytes + 63) & ~size_t(63);
  if (use_pinned_staging) {
    if (routed_count > SIZE_MAX / bytes_per_active_expert ||
        packed_offset >
            SIZE_MAX - routed_count * bytes_per_active_expert) {
      fprintf(stderr, "wrap_qmoe_from_file: pinned staging size overflow\n");
      return -1;
    }
    const size_t pinned_bytes =
        packed_offset + routed_count * bytes_per_active_expert;
    if (hipdnn_ep_state_ensure_qmoe_host_scratch(state, pinned_bytes) != 0) {
      fprintf(stderr,
              "wrap_qmoe_from_file: unable to allocate pinned expert staging\n");
      return -1;
    }
    host_scratch = static_cast<uint8_t *>(
        hipdnn_ep_state_get_qmoe_host_scratch(state));
  }

  std::vector<int32_t> pageable_active_experts;
  int32_t *active_data = nullptr;
  if (use_pinned_staging) {
    active_data = reinterpret_cast<int32_t *>(host_scratch);
  } else {
    pageable_active_experts.resize(routed_count);
    active_data = pageable_active_experts.data();
  }
  hipError_t status = hipMemcpyAsync(active_data, route_indices,
                                     route_indices_bytes,
                                     hipMemcpyDeviceToHost, stream);
  if (status != hipSuccess || hipStreamSynchronize(stream) != hipSuccess) {
    fprintf(stderr, "wrap_qmoe_from_file: unable to read active experts\n");
    return -1;
  }
  std::vector<int32_t> active_experts(active_data,
                                      active_data + routed_count);
  std::sort(active_experts.begin(), active_experts.end());
  active_experts.erase(
      std::unique(active_experts.begin(), active_experts.end()),
      active_experts.end());

  // Decode always remaps the selected experts into compact slots. Without a
  // cache the slots are [0, active_experts); with a cache they address a small
  // per-layer LRU allocation. This avoids the old cache's full num_experts
  // sparse allocation while retaining already-uploaded expert slices.
  const bool compact_decode = num_tokens == 1;
  size_t device_table_offsets[4] = {};
  size_t device_total_size = 0;
  for (int i = 0; i < 4; ++i) {
    device_total_size =
        (device_total_size + kTableAlignment - 1) & ~(kTableAlignment - 1);
    device_table_offsets[i] = device_total_size;
    const size_t device_table_size = compact_decode
                                         ? active_experts.size() * expert_sizes[i]
                                         : table_sizes[i];
    if (device_table_size > SIZE_MAX - device_total_size) {
      fprintf(stderr, "wrap_qmoe_from_file: compact table overflow\n");
      return -1;
    }
    device_total_size += device_table_size;
  }

  uint8_t *device_base = nullptr;
  if (cache_entry) {
    device_base = static_cast<uint8_t *>(cache_entry->device_base);
    device_total_size = cache_entry->total_size;
    for (int i = 0; i < 4; ++i)
      device_table_offsets[i] = cache_entry->table_offsets[i];
  } else {
    if (hipdnn_ep_state_ensure_workspace(state, device_total_size) != 0) {
      fprintf(stderr,
              "wrap_qmoe_from_file: unable to allocate %zu-byte GPU staging "
              "buffer\n",
              device_total_size);
      return -1;
    }
    device_base =
        static_cast<uint8_t *>(hipdnn_ep_state_get_workspace(state));
  }
  if (!device_base) {
    fprintf(stderr, "wrap_qmoe_from_file: missing device staging\n");
    return -1;
  }
  void *device_tables[4] = {};
  for (int i = 0; i < 4; ++i)
    device_tables[i] = device_base + device_table_offsets[i];

  size_t streamed_bytes = 0;
  size_t packed_cursor = packed_offset;
  size_t cache_hits = 0;
  std::vector<int32_t> active_slots(active_experts.size(), -1);
  std::vector<int32_t> miss_experts;
  std::vector<int32_t> miss_slots;
  if (compact_decode) {
    for (size_t active = 0; active < active_experts.size(); ++active) {
      const int32_t expert = active_experts[active];
      if (expert < 0 || expert >= num_experts) {
        fprintf(stderr,
                "wrap_qmoe_from_file: invalid routed expert %d of %lld\n",
                expert, (long long)num_experts);
        return -1;
      }
      int32_t slot = -1;
      if (cache_entry) {
        for (size_t candidate = 0; candidate < cache_entry->capacity;
             ++candidate) {
          if (cache_entry->expert_for_slot[candidate] == expert) {
            slot = static_cast<int32_t>(candidate);
            break;
          }
        }
        if (slot >= 0) {
          ++cache_hits;
        } else {
          for (size_t candidate = 0; candidate < cache_entry->capacity;
               ++candidate) {
            if (cache_entry->expert_for_slot[candidate] == -1) {
              slot = static_cast<int32_t>(candidate);
              break;
            }
          }
          if (slot < 0) {
            size_t least_used = SIZE_MAX;
            size_t oldest = SIZE_MAX;
            for (size_t candidate = 0; candidate < cache_entry->capacity;
                 ++candidate) {
              const int32_t resident =
                  cache_entry->expert_for_slot[candidate];
              if (std::binary_search(active_experts.begin(),
                                     active_experts.end(), resident))
                continue;
              const size_t candidate_use = cache_entry->use_count[candidate];
              const size_t candidate_age = cache_entry->last_used[candidate];
              if (slot < 0 || candidate_use < least_used ||
                  (candidate_use == least_used && candidate_age < oldest)) {
                slot = static_cast<int32_t>(candidate);
                least_used = candidate_use;
                oldest = candidate_age;
              }
            }
          }
          if (slot < 0) {
            fprintf(stderr, "wrap_qmoe_from_file: no evictable cache slot\n");
            return -1;
          }
          cache_entry->expert_for_slot[static_cast<size_t>(slot)] = expert;
          cache_entry->use_count[static_cast<size_t>(slot)] = 0;
          miss_experts.push_back(expert);
          miss_slots.push_back(slot);
        }
        ++cache_entry->use_count[static_cast<size_t>(slot)];
        cache_entry->last_used[static_cast<size_t>(slot)] =
            ++cache_entry->tick;
      } else {
        slot = static_cast<int32_t>(active);
        miss_experts.push_back(expert);
        miss_slots.push_back(slot);
      }
      active_slots[active] = slot;
    }

    for (size_t j = 0; j < routed_count; ++j) {
      auto it = std::lower_bound(active_experts.begin(), active_experts.end(),
                                 active_data[j]);
      if (it == active_experts.end() || *it != active_data[j]) {
        fprintf(stderr, "wrap_qmoe_from_file: route remap failed\n");
        return -1;
      }
      active_data[j] = active_slots[static_cast<size_t>(
          it - active_experts.begin())];
    }
    status = hipMemcpyAsync(route_indices, active_data, route_indices_bytes,
                            hipMemcpyHostToDevice, stream);
    if (status != hipSuccess) {
      fprintf(stderr, "wrap_qmoe_from_file: route remap upload failed\n");
      return -1;
    }

    std::vector<size_t> miss_order;
    miss_order.reserve(miss_slots.size());
    for (size_t i = 0; i < miss_slots.size(); ++i)
      miss_order.push_back(i);
    std::sort(miss_order.begin(), miss_order.end(), [&](size_t lhs, size_t rhs) {
      return miss_slots[lhs] < miss_slots[rhs];
    });

    for (int i = 0; i < 4; ++i) {
      const size_t expert_bytes = expert_sizes[i];
      size_t begin = 0;
      while (begin < miss_order.size()) {
        size_t end = begin + 1;
        while (end < miss_order.size() &&
               miss_slots[miss_order[end]] ==
                   miss_slots[miss_order[end - 1]] + 1)
          ++end;
        const size_t run_count = end - begin;
        const size_t packed_run_bytes = run_count * expert_bytes;
        uint8_t *packed_run = host_scratch + packed_cursor;
        for (size_t run = 0; run < run_count; ++run) {
          const size_t miss = miss_order[begin + run];
          const size_t expert = static_cast<size_t>(miss_experts[miss]);
          std::memcpy(packed_run + run * expert_bytes,
                      static_cast<const uint8_t *>(host_tables[i]) +
                          expert * expert_bytes,
                      expert_bytes);
        }
        const size_t first_slot =
            static_cast<size_t>(miss_slots[miss_order[begin]]);
        status = hipMemcpyAsync(
            static_cast<uint8_t *>(device_tables[i]) +
                first_slot * expert_bytes,
            packed_run, packed_run_bytes, hipMemcpyHostToDevice, stream);
        if (status != hipSuccess) {
          fprintf(stderr,
                  "wrap_qmoe_from_file: compact H2D failed for constant %lld "
                  "(%zu bytes): %s\n",
                  (long long)indices[i], packed_run_bytes,
                  hipGetErrorString(status));
          return -1;
        }
        packed_cursor += packed_run_bytes;
        streamed_bytes += packed_run_bytes;
        begin = end;
      }
    }
  } else {
    for (int32_t expert : active_experts) {
      if (expert < 0 || expert >= num_experts) {
        fprintf(stderr,
                "wrap_qmoe_from_file: invalid routed expert %d of %lld\n",
                expert, (long long)num_experts);
        return -1;
      }
      for (int i = 0; i < 4; ++i) {
        const size_t expert_bytes = expert_sizes[i];
        const size_t expert_offset = static_cast<size_t>(expert) * expert_bytes;
        status = hipMemcpyAsync(static_cast<uint8_t *>(device_tables[i]) +
                                    expert_offset,
                                static_cast<const uint8_t *>(host_tables[i]) +
                                    expert_offset,
                                expert_bytes, hipMemcpyHostToDevice, stream);
        if (status != hipSuccess) {
          fprintf(stderr,
                  "wrap_qmoe_from_file: H2D copy failed for constant %lld "
                  "expert %d (%zu bytes): %s\n",
                  (long long)indices[i], expert, expert_bytes,
                  hipGetErrorString(status));
          return -1;
        }
        streamed_bytes += expert_bytes;
      }
    }
  }
  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_qmoe_from_file: streamed %zu bytes for %zu active experts "
      "across four tables (%zu cache hits, compact=%d)\n",
      streamed_bytes, active_experts.size(), cache_hits,
      compact_decode ? 1 : 0);

  if (compact_decode) {
    hipdnn_ep_state_set_qmoe_precomputed_routes(state, route_indices,
                                                route_weights);
  }
  const int result = wrap_qmoe(
      state, input, router_probs, router_weights, device_tables[0],
      device_tables[1], fc1_bias, device_tables[2], device_tables[3],
      fc2_bias, fc3_weights, fc3_scales, fc3_bias, fc1_zero_points,
      fc2_zero_points, fc3_zero_points, output, num_tokens, hidden_size,
      inter_size, num_experts, k, expert_weight_bits, block_size,
      swiglu_fusion, activation_type, activation_alpha, activation_beta,
      swiglu_limit, normalize_routing_weights, elem_size);
  hipdnn_ep_state_set_qmoe_precomputed_routes(state, nullptr, nullptr);
  return result;
}

int wrap_qmoe(RuntimeState *state, const void *input, const void *router_probs,
              const void *router_weights, const void *fc1_weights,
              const void *fc1_scales, const void *fc1_bias,
              const void *fc2_weights, const void *fc2_scales,
              const void *fc2_bias, const void *fc3_weights,
              const void *fc3_scales, const void *fc3_bias,
              const void *fc1_zero_points, const void *fc2_zero_points,
              const void *fc3_zero_points, void *output, int64_t num_tokens,
              int64_t hidden_size, int64_t inter_size, int64_t num_experts,
              int64_t k, int64_t expert_weight_bits, int64_t block_size,
              int64_t swiglu_fusion, int64_t activation_type,
              float activation_alpha, float activation_beta, float swiglu_limit,
              int64_t normalize_routing_weights, int64_t elem_size) {
  OP_PROFILE(
      "qmoe",
      [&] {
        char b[64];
        snprintf(b, sizeof(b), "%lldx%lldx%lld,e=%lld", (long long)num_tokens,
                 (long long)hidden_size, (long long)inter_size,
                 (long long)num_experts);
        return std::string(b);
      },
      state);
  if (router_weights) {
    fprintf(stderr, "wrap_qmoe: router_weights is not supported yet\n");
    return -1;
  }
  if (!state || !input || !router_probs || !output) {
    fprintf(stderr, "wrap_qmoe: null argument\n");
    return -1;
  }

  if (swiglu_fusion != 1) {
    fprintf(stderr, "wrap_qmoe: only swiglu_fusion=1 supported, got %lld\n",
            (long long)swiglu_fusion);
    return -1;
  }

  if (fc3_weights || fc3_scales || fc3_bias || fc3_zero_points) {
    fprintf(stderr, "wrap_qmoe: fc3 (unfused SwiGLU) not supported, "
                    "use swiglu_fusion=1\n");
    return -1;
  }

  RUNTIME_DEBUG_LOG("[REAL] wrap_qmoe(tokens=%lld, hidden=%lld, inter=%lld, "
                    "experts=%lld, k=%lld, bits=%lld, block=%lld, elem=%lld)\n",
                    (long long)num_tokens, (long long)hidden_size,
                    (long long)inter_size, (long long)num_experts, (long long)k,
                    (long long)expert_weight_bits, (long long)block_size,
                    (long long)elem_size);

  // Guard against pathological metadata: block_size==0 would otherwise crash
  // with STATUS_INTEGER_DIVIDE_BY_ZERO inside the k_blocks computations below
  // (and produces invalid quant layouts even at >0 if not a multiple of 2).
  if (block_size <= 0 || (block_size & 1) != 0) {
    fprintf(stderr,
            "wrap_qmoe: invalid block_size=%lld (must be a positive even "
            "value matching the weights' quant block layout)\n",
            (long long)block_size);
    return -1;
  }
  if (hidden_size <= 0 || inter_size <= 0 || num_experts <= 0 || k <= 0 ||
      num_tokens <= 0 || elem_size <= 0) {
    fprintf(stderr,
            "wrap_qmoe: invalid sizes (tokens=%lld hidden=%lld inter=%lld "
            "experts=%lld k=%lld elem=%lld)\n",
            (long long)num_tokens, (long long)hidden_size,
            (long long)inter_size, (long long)num_experts, (long long)k,
            (long long)elem_size);
    return -1;
  }

  void *stream = hipdnn_ep_state_get_stream(state);
  if (!stream) {
    fprintf(stderr, "wrap_qmoe: null stream\n");
    return -1;
  }

  int result = 0;

  int64_t fusion_inter = 2 * inter_size;
  int64_t k_blocks_fc1 = (hidden_size + block_size - 1) / block_size;
  int64_t k_blocks_fc2 = (inter_size + block_size - 1) / block_size;
  int64_t routed_rows = num_tokens * k;

  // Grouped prefill: fc1/fc2 run as two direct-B small-M WMMA launches each
  // over a device-built row-tile list (experts of <= 16 rows as one 16-row
  // tile, larger ones as 32-row tiles) instead of the ragged persistent GEMM.
  // gfx1151, gpt-oss-120b routing of a 2k prompt in 512-token chunks: 11.8 vs
  // 24.4 ms per layer. Shapes the small-M kernel cannot run use the ragged
  // GEMM.
  constexpr int64_t kGroupedSplitRows = 16;
  constexpr int64_t kGroupedSmallTile = 16;
  constexpr int64_t kGroupedBigTile = 32;
  auto grouped_fits = [&](int64_t tile_rows) {
    return hip_matmul_nbits_wmma_smallm_grouped_supported(
               -1, tile_rows, fusion_inter, hidden_size, expert_weight_bits,
               block_size, elem_size) &&
           hip_matmul_nbits_wmma_smallm_grouped_supported(
               -1, tile_rows, hidden_size, inter_size, expert_weight_bits,
               block_size, elem_size);
  };
  const bool grouped = num_tokens > 1 && grouped_fits(kGroupedSmallTile) &&
                       grouped_fits(kGroupedBigTile);
  // Tile-list capacities (see hip_qmoe_build_row_tiles): a small expert is one
  // tile; big experts need routed_rows / 32 plus one partial tile each.
  const int64_t max_small_tiles = std::min(num_experts, routed_rows);
  const int64_t max_big_tiles =
      routed_rows / kGroupedBigTile +
      std::min(num_experts, routed_rows / (kGroupedSplitRows + 1));

  // Per-state grow-on-demand scratch in place of 8 hipMalloc/8 hipFree per
  // call. Sub-buffers are 64-byte aligned (matches GPU pool alignment, gives
  // each sub-buffer its own cache line). The buffer grows when num_tokens /
  // sizes exceed the cached capacity, never shrinks; freed in state cleanup.
  auto align_up_64 = [](size_t s) -> size_t { return (s + 63) & ~size_t(63); };
  size_t sz_expert_indices = align_up_64(num_tokens * k * sizeof(int32_t));
  size_t sz_expert_weights = align_up_64(num_tokens * k * elem_size);
  size_t sz_fc1_buf = align_up_64(routed_rows * fusion_inter * elem_size);
  // Fused decode (num_tokens == 1) reuses act_buf and fc2_buf as the [k,
  // inter] activation slots and [k, hidden] per-expert output slots needed
  // by hip_qmoe_decode_fused (gather/scatter happen inline inside the
  // kernel, indexed by expert_indices). For num_tokens > 1 the multi-pass
  // path uses [num_tokens, ...] sizing. Take the max so the per-state
  // scratch is never under-sized regardless of which path runs.
  int64_t act_slots = std::max<int64_t>(routed_rows, k);
  size_t sz_act_buf = align_up_64(act_slots * inter_size * elem_size);
  size_t sz_fc2_buf = align_up_64(act_slots * hidden_size * elem_size);
  // Device bucket outputs: per-expert counts/offsets, sorted routed pairs,
  // inverse pair mapping, adaptive <=64-row task descriptors, and two queues.
  size_t sz_expert_counts = align_up_64(num_experts * sizeof(int32_t));
  size_t sz_expert_offsets = align_up_64((num_experts + 1) * sizeof(int32_t));
  size_t sz_sorted_token_ids = align_up_64(num_tokens * k * sizeof(int32_t));
  size_t sz_sorted_weights = align_up_64(num_tokens * k * elem_size);
  size_t sz_pair_to_sorted = align_up_64(routed_rows * sizeof(int32_t));
  size_t sz_row_groups = align_up_64(routed_rows * 4 * sizeof(int32_t));
  size_t sz_row_group_count = align_up_64(sizeof(int32_t));
  size_t sz_fc1_queue_head = align_up_64(sizeof(uint32_t));
  size_t sz_fc2_queue_head = align_up_64(sizeof(uint32_t));
  // dp4a decode scratch (fused decode only, env-gated): int8-quantized
  // activations + per-group fp32 scales for the fc1 input ([hidden]) and the
  // fc2 slot activations ([k, inter]). Sized unconditionally (a few KB) so the
  // offset layout is identical whether or not dp4a runs; the fp path ignores
  // them. k_blocks_fc1 == n_blk_in, k_blocks_fc2 == n_blk_mid.
  size_t sz_a_qb_in = align_up_64(hidden_size * sizeof(int8_t));
  size_t sz_a_scale_in = align_up_64(k_blocks_fc1 * sizeof(float));
  size_t sz_a_qb_mid = align_up_64(k * inter_size * sizeof(int8_t));
  size_t sz_a_scale_mid = align_up_64(k * k_blocks_fc2 * sizeof(float));
  // Grouped prefill only: the gathered fc1 input rows, the two row-tile lists
  // (3 int32 per tile) and their device-side lengths.
  size_t sz_gather_buf =
      grouped ? align_up_64(routed_rows * hidden_size * elem_size) : 0;
  size_t sz_tiles_small =
      grouped ? align_up_64(max_small_tiles * 3 * sizeof(int32_t)) : 0;
  size_t sz_tiles_big =
      grouped ? align_up_64(max_big_tiles * 3 * sizeof(int32_t)) : 0;
  size_t sz_num_tiles = grouped ? align_up_64(2 * sizeof(int32_t)) : 0;

  size_t off_expert_indices = 0;
  size_t off_expert_weights = off_expert_indices + sz_expert_indices;
  size_t off_fc1_buf = off_expert_weights + sz_expert_weights;
  size_t off_act_buf = off_fc1_buf + sz_fc1_buf;
  size_t off_fc2_buf = off_act_buf + sz_act_buf;
  size_t off_expert_counts = off_fc2_buf + sz_fc2_buf;
  size_t off_expert_offsets = off_expert_counts + sz_expert_counts;
  size_t off_sorted_token_ids = off_expert_offsets + sz_expert_offsets;
  size_t off_sorted_weights = off_sorted_token_ids + sz_sorted_token_ids;
  size_t off_pair_to_sorted = off_sorted_weights + sz_sorted_weights;
  size_t off_row_groups = off_pair_to_sorted + sz_pair_to_sorted;
  size_t off_row_group_count = off_row_groups + sz_row_groups;
  size_t off_fc1_queue_head = off_row_group_count + sz_row_group_count;
  size_t off_fc2_queue_head = off_fc1_queue_head + sz_fc1_queue_head;
  size_t off_a_qb_in = off_fc2_queue_head + sz_fc2_queue_head;
  size_t off_a_scale_in = off_a_qb_in + sz_a_qb_in;
  size_t off_a_qb_mid = off_a_scale_in + sz_a_scale_in;
  size_t off_a_scale_mid = off_a_qb_mid + sz_a_qb_mid;
  size_t off_gather_buf = off_a_scale_mid + sz_a_scale_mid;
  size_t off_tiles_small = off_gather_buf + sz_gather_buf;
  size_t off_tiles_big = off_tiles_small + sz_tiles_small;
  size_t off_num_tiles = off_tiles_big + sz_tiles_big;
  size_t total_scratch = off_num_tiles + sz_num_tiles;

  if (hipdnn_ep_state_ensure_qmoe_scratch(state, total_scratch) != 0) {
    fprintf(stderr, "wrap_qmoe: ensure_qmoe_scratch(%zu) failed\n",
            total_scratch);
    return -1;
  }
  char *scratch_base =
      static_cast<char *>(hipdnn_ep_state_get_qmoe_scratch(state));
  void *d_expert_indices = scratch_base + off_expert_indices;
  void *d_expert_weights = scratch_base + off_expert_weights;
  void *d_fc1_buf = scratch_base + off_fc1_buf;
  void *d_act_buf = scratch_base + off_act_buf;
  void *d_fc2_buf = scratch_base + off_fc2_buf;
  int32_t *d_expert_counts =
      reinterpret_cast<int32_t *>(scratch_base + off_expert_counts);
  int32_t *d_expert_offsets =
      reinterpret_cast<int32_t *>(scratch_base + off_expert_offsets);
  int32_t *d_sorted_token_ids =
      reinterpret_cast<int32_t *>(scratch_base + off_sorted_token_ids);
  char *d_sorted_weights = scratch_base + off_sorted_weights;
  int32_t *d_pair_to_sorted =
      reinterpret_cast<int32_t *>(scratch_base + off_pair_to_sorted);
  int32_t *d_row_groups =
      reinterpret_cast<int32_t *>(scratch_base + off_row_groups);
  int32_t *d_row_group_count =
      reinterpret_cast<int32_t *>(scratch_base + off_row_group_count);
  uint32_t *d_fc1_queue_head =
      reinterpret_cast<uint32_t *>(scratch_base + off_fc1_queue_head);
  uint32_t *d_fc2_queue_head =
      reinterpret_cast<uint32_t *>(scratch_base + off_fc2_queue_head);
  void *d_a_qb_in = scratch_base + off_a_qb_in;
  void *d_a_scale_in = scratch_base + off_a_scale_in;
  void *d_a_qb_mid = scratch_base + off_a_qb_mid;
  void *d_a_scale_mid = scratch_base + off_a_scale_mid;
  void *d_gather_buf = scratch_base + off_gather_buf;
  void *d_tiles_small = scratch_base + off_tiles_small;
  void *d_tiles_big = scratch_base + off_tiles_big;
  int32_t *d_num_tiles =
      reinterpret_cast<int32_t *>(scratch_base + off_num_tiles);

  RUNTIME_DEBUG_LOG("[REAL] wrap_qmoe: topk_routing(tokens=%lld, experts=%lld, "
                    "k=%lld, normalize=%lld)\n",
                    (long long)num_tokens, (long long)num_experts, (long long)k,
                    (long long)normalize_routing_weights);
  const bool reuse_precomputed =
      hipdnn_ep_state_get_qmoe_precomputed_indices(state) ==
          d_expert_indices &&
      hipdnn_ep_state_get_qmoe_precomputed_weights(state) ==
          d_expert_weights;
  const bool expected_precomputed =
      hipdnn_ep_state_get_qmoe_precomputed_indices(state) != nullptr;
  if (expected_precomputed && !reuse_precomputed) {
    fprintf(stderr,
            "wrap_qmoe: compact routing scratch moved unexpectedly\n");
    return -1;
  }
  if (!reuse_precomputed) {
    HIP_CHECK(hip_qmoe_topk_routing(stream, router_probs, d_expert_indices,
                                    d_expert_weights, num_tokens, num_experts,
                                    k, normalize_routing_weights, elem_size));
  } else {
    RUNTIME_DEBUG_LOG("[REAL] wrap_qmoe: reusing precomputed compact routes\n");
  }

  // Fused decode fast path: single-token MoE collapses to three back-to-back
  // kernel launches (FC1+SwiGLU, FC2, weighted reduce) with zero D2H,
  // hipStreamSynchronize, or host-side bucketing. Replaces the multi-pass
  // bucket -> sync -> per-active-expert (gather, fc1, swiglu, fc2,
  // scatter_add) sequence below. d_act_buf is reused as the [k, inter]
  // activation slots, d_fc2_buf as the [k, hidden] per-expert output slots
  // (gather/scatter happen inline via expert_indices).
  if (num_tokens == 1) {
    // W4A8 dp4a decode variant (env-gated). Requires fp16 and 32-aligned
    // block_size / hidden / inter (all true for the MoE targets: block_size
    // 32, hidden/inter multiples of 32). Quantizes the shared input + the k
    // slot activations to int8 once, then runs the fc1/fc2 GEMVs via sudot4.
    // Falls through to the fp fused path otherwise.
    const bool dp4a_ok = hipdnn_ep_matmul_dp4a_enabled() && elem_size == 2 &&
                         block_size > 0 && (block_size % 32 == 0) &&
                         (hidden_size % 32 == 0) && (inter_size % 32 == 0);
    if (dp4a_ok) {
      RUNTIME_DEBUG_LOG("[REAL] wrap_qmoe: fused decode dp4a path (k=%lld)\n",
                        (long long)k);
      HIP_CHECK(hip_qmoe_decode_fused_dp4a(
          stream, input, d_expert_indices, d_expert_weights, fc1_weights,
          fc1_scales, fc1_zero_points, fc1_bias, fc2_weights, fc2_scales,
          fc2_zero_points, fc2_bias, d_fc2_buf, d_act_buf, output, d_a_qb_in,
          d_a_scale_in, d_a_qb_mid, d_a_scale_mid, hidden_size, inter_size, k,
          block_size, activation_alpha, activation_beta, swiglu_limit,
          elem_size));
      return 0;
    }
    RUNTIME_DEBUG_LOG("[REAL] wrap_qmoe: fused decode path (k=%lld)\n",
                      (long long)k);
    HIP_CHECK(hip_qmoe_decode_fused(
        stream, input, d_expert_indices, d_expert_weights, fc1_weights,
        fc1_scales, fc1_zero_points, fc1_bias, fc2_weights, fc2_scales,
        fc2_zero_points, fc2_bias, d_fc2_buf, d_act_buf, output, hidden_size,
        inter_size, k, block_size, activation_alpha, activation_beta,
        swiglu_limit, elem_size));
    return 0;
  }

  // Device-driven prefill path. The bucket launch creates every descriptor
  // consumed below and resets two independent persistent queues. FC1 gathers
  // source token rows through d_sorted_token_ids; FC2 consumes the contiguous
  // sorted activation rows. No expert metadata is copied to the host.
  {
    if (expert_weight_bits != 4) {
      fprintf(stderr,
              "wrap_qmoe: ragged prefill supports expert_weight_bits=4, got "
              "%lld\n",
              (long long)expert_weight_bits);
      return -1;
    }

    // The small-M kernel reads the packed zero points directly; only the
    // ragged GEMM needs them unpacked.
    const void *fc1_pre_zp_u8 = nullptr;
    const void *fc2_pre_zp_u8 = nullptr;
    if (!grouped && (fc1_zero_points || fc2_zero_points)) {
      hipdnn_ep_real::ZpUnpackCache *zpc =
          hipdnn_ep_real::get_or_create_zp_cache(state);
      if (!zpc) {
        fprintf(stderr, "wrap_qmoe: failed to create zero-point cache\n");
        return -1;
      }
      if (fc1_zero_points) {
        fc1_pre_zp_u8 = hipdnn_ep_real::lookup_or_unpack_zp_u8(
            *zpc, stream, fc1_zero_points,
            static_cast<int>(num_experts * fusion_inter),
            static_cast<int>(k_blocks_fc1));
        if (!fc1_pre_zp_u8) {
          result = -1;
          goto cleanup;
        }
      }
      if (fc2_zero_points) {
        fc2_pre_zp_u8 = hipdnn_ep_real::lookup_or_unpack_zp_u8(
            *zpc, stream, fc2_zero_points,
            static_cast<int>(num_experts * hidden_size),
            static_cast<int>(k_blocks_fc2));
        if (!fc2_pre_zp_u8) {
          result = -1;
          goto cleanup;
        }
      }
    }

    constexpr int64_t row_group_size = 64;
    RUNTIME_DEBUG_LOG(
        "[REAL] wrap_qmoe: device %s prefill rows=%lld experts=%lld\n",
        grouped ? "grouped" : "ragged", (long long)routed_rows,
        (long long)num_experts);
    if (grouped) {
      // Routing slots that name no expert leave the tail of the sorted ids
      // unwritten; zero it so the fixed-size gather below stays in bounds.
      HIP_CHECK(hipMemsetAsync(d_sorted_token_ids, 0,
                               routed_rows * sizeof(int32_t),
                               static_cast<hipStream_t>(stream)));
    }
    HIP_CHECK(hip_qmoe_bucket_tokens_ragged(
        stream, d_expert_indices, d_expert_weights, d_expert_counts,
        d_expert_offsets, d_sorted_token_ids, d_sorted_weights,
        d_pair_to_sorted, d_row_groups, d_row_group_count, d_fc1_queue_head,
        d_fc2_queue_head, num_tokens, num_experts, k, row_group_size,
        elem_size));

    if (grouped) {
      HIP_CHECK(hip_qmoe_build_row_tiles(
          stream, d_expert_counts, num_experts, kGroupedSplitRows,
          kGroupedSmallTile, kGroupedBigTile, d_tiles_small, d_num_tiles,
          d_tiles_big, d_num_tiles + 1));
      HIP_CHECK(hip_qmoe_gather_tokens(stream, input, d_gather_buf,
                                       d_sorted_token_ids, hidden_size,
                                       routed_rows, elem_size));

      auto grouped_fc = [&](const void *a, const void *w, const void *s,
                            const void *zp, const void *b, void *out, int64_t n,
                            int64_t kdim) -> int {
        int rc = hip_matmul_nbits_wmma_smallm_grouped(
            stream, -1, a, w, s, zp, /*zp_packed_u4=*/1, b, d_expert_offsets,
            d_tiles_small, d_num_tiles, out, max_small_tiles, kGroupedSmallTile,
            n, kdim, expert_weight_bits, block_size, elem_size);
        if (rc != 0)
          return rc;
        return hip_matmul_nbits_wmma_smallm_grouped(
            stream, -1, a, w, s, zp, /*zp_packed_u4=*/1, b, d_expert_offsets,
            d_tiles_big, d_num_tiles + 1, out, max_big_tiles, kGroupedBigTile,
            n, kdim, expert_weight_bits, block_size, elem_size);
      };
      HIP_CHECK(grouped_fc(d_gather_buf, fc1_weights, fc1_scales,
                           fc1_zero_points, fc1_bias, d_fc1_buf, fusion_inter,
                           hidden_size));
      HIP_CHECK(hip_qmoe_swiglu(stream, d_fc1_buf, d_act_buf, routed_rows,
                                inter_size, activation_alpha, activation_beta,
                                swiglu_limit, elem_size));
      HIP_CHECK(grouped_fc(d_act_buf, fc2_weights, fc2_scales, fc2_zero_points,
                           fc2_bias, d_fc2_buf, hidden_size, inter_size));
      HIP_CHECK(hip_qmoe_reduce_sorted_pairs(
          stream, d_fc2_buf, d_pair_to_sorted, d_expert_indices,
          d_expert_weights, output, num_tokens, hidden_size, k, elem_size));
      return 0;
    }

    HIP_CHECK(hip_qmoe_ragged_matmul_nbits(
        stream, input, d_sorted_token_ids, d_row_groups, d_row_group_count,
        d_fc1_queue_head, fc1_weights, fc1_scales, fc1_pre_zp_u8, fc1_bias,
        d_fc1_buf, routed_rows, num_experts, fusion_inter, hidden_size,
        block_size, elem_size));

    HIP_CHECK(hip_qmoe_swiglu(stream, d_fc1_buf, d_act_buf, routed_rows,
                              inter_size, activation_alpha, activation_beta,
                              swiglu_limit, elem_size));

    HIP_CHECK(hip_qmoe_ragged_matmul_nbits(
        stream, d_act_buf, /*input_row_ids=*/nullptr, d_row_groups,
        d_row_group_count, d_fc2_queue_head, fc2_weights, fc2_scales,
        fc2_pre_zp_u8, fc2_bias, d_fc2_buf, routed_rows, num_experts,
        hidden_size, inter_size, block_size, elem_size));

    HIP_CHECK(hip_qmoe_reduce_sorted_pairs(
        stream, d_fc2_buf, d_pair_to_sorted, d_expert_indices, d_expert_weights,
        output, num_tokens, hidden_size, k, elem_size));
    return 0;
  }

cleanup:
  // Sub-buffers above (d_expert_indices ... d_sorted_weights) are views into
  // the per-session RuntimeState::qmoe_scratch pool -- freed in
  // hipdnn_ep_state_cleanup. Do NOT hipFree them here.
  if (result == 0) {
    RUNTIME_DEBUG_LOG("[REAL] wrap_qmoe: completed successfully\n");
  }
  return result;
}
