#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <new>

namespace {

constexpr uint32_t kPairTile = 16;
constexpr uint32_t kWordTile = 16;
constexpr uint32_t kBatchRows = 256;
constexpr uint32_t kBatchPairs = kBatchRows * kBatchRows;

struct CandidatePair {
  uint32_t left;
  uint32_t right;
};

struct FilterStatistics {
  uint64_t launches = 0;
  uint64_t examined_pairs = 0;
  uint64_t accepted_pairs = 0;
  uint64_t upload_bytes = 0;
  uint64_t download_bytes = 0;
  uint64_t device_bytes = 0;
  double upload_ms = 0;
  double kernel_ms = 0;
  double download_ms = 0;
  double allocation_ms = 0;
};

struct NativeError {
  int code;
  char message[256];
};

static_assert(sizeof(CandidatePair) == 8);
static_assert(sizeof(FilterStatistics) == 80);

int failure(NativeError* error, int code, const char* message) noexcept {
  error->code = code;
  std::snprintf(error->message, sizeof(error->message), "%s", message);
  return -1;
}

int cuda_failure(NativeError* error, const char* operation, cudaError_t status) noexcept {
  error->code = static_cast<int>(status);
  std::snprintf(error->message, sizeof(error->message), "%s: %s", operation,
                cudaGetErrorString(status));
  return -1;
}

#define CUDA_TRY(operation)                                    \
  do {                                                         \
    const cudaError_t status = (operation);                     \
    if (status != cudaSuccess) {                               \
      return cuda_failure(error, #operation, status);           \
    }                                                          \
  } while (false)

struct Round {
  int device = 0;
  uint32_t row_count = 0;
  uint32_t word_count = 0;
  uint32_t min_chunks = 1;
  uint32_t row_begin = 0;
  uint32_t column_begin = 0;
  cudaStream_t stream = nullptr;
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  uint64_t* words = nullptr;
  uint32_t* lengths = nullptr;
  uint64_t* excluded = nullptr;
  CandidatePair* pairs = nullptr;
  uint32_t* count = nullptr;
  FilterStatistics statistics;

  ~Round() noexcept {
    // Every successful batch is synchronized. A failed asynchronous operation
    // can still leave work pending, so synchronize before releasing its buffers.
    cudaSetDevice(device);
    if (stream != nullptr) cudaStreamSynchronize(stream);
    if (words != nullptr) cudaFree(words);
    if (lengths != nullptr) cudaFree(lengths);
    if (excluded != nullptr) cudaFree(excluded);
    if (pairs != nullptr) cudaFree(pairs);
    if (count != nullptr) cudaFree(count);
    if (start != nullptr) cudaEventDestroy(start);
    if (stop != nullptr) cudaEventDestroy(stop);
    if (stream != nullptr) cudaStreamDestroy(stream);
  }
};

int finish_interval(Round& round, double& milliseconds, NativeError* error) noexcept {
  CUDA_TRY(cudaEventRecord(round.stop, round.stream));
  CUDA_TRY(cudaEventSynchronize(round.stop));
  float elapsed = 0;
  CUDA_TRY(cudaEventElapsedTime(&elapsed, round.start, round.stop));
  milliseconds += elapsed;
  return 0;
}

// One block filters 16 x 16 pairs. Cooperative loads reuse each input word for
// sixteen comparisons. The transposed right tile keeps simultaneous shared
// memory reads adjacent instead of making all lanes read the same bank.
__global__ void filter_pairs(const uint64_t* words, const uint32_t* lengths,
                             const uint64_t* excluded, uint32_t word_count,
                             uint32_t min_chunks, uint32_t row_begin,
                             uint32_t row_end, uint32_t column_begin,
                             uint32_t column_end, CandidatePair* output,
                             uint32_t* output_count) {
  __shared__ uint64_t left_tile[kPairTile][kWordTile];
  __shared__ uint64_t right_tile[kWordTile][kPairTile];
  __shared__ uint64_t excluded_tile[kWordTile];

  const uint32_t local_column = threadIdx.x;
  const uint32_t local_row = threadIdx.y;
  const uint32_t block_row = row_begin + blockIdx.y * kPairTile;
  const uint32_t block_column = column_begin + blockIdx.x * kPairTile;
  const uint64_t row = static_cast<uint64_t>(block_row) + local_row;
  const uint64_t column = static_cast<uint64_t>(block_column) + local_column;
  bool eligible = row < row_end && column < column_end && column < row;
  if (eligible) {
    eligible = lengths[row] > min_chunks && lengths[column] > min_chunks;
  }
  uint64_t count = 0;
  uint64_t common = 0;
  uint64_t left_difference = 0;
  uint64_t right_difference = 0;

  for (uint64_t base = 0; base < word_count; base += kWordTile) {
    const uint64_t word = base + local_column;
    const uint32_t tile_words = static_cast<uint32_t>(
      word_count - base < kWordTile ? word_count - base : kWordTile);
    const uint64_t right_row = static_cast<uint64_t>(block_column) + local_row;
    left_tile[local_row][local_column] = row < row_end && word < word_count
      ? words[static_cast<uint64_t>(row) * word_count + word] : 0;
    right_tile[local_column][local_row] = right_row < column_end && word < word_count
      ? words[static_cast<uint64_t>(right_row) * word_count + word] : 0;
    if (local_row == 0) {
      excluded_tile[local_column] = excluded != nullptr && word < word_count
        ? excluded[word] : 0;
    }
    __syncthreads();

    if (eligible) {
      for (uint32_t offset = 0; offset < tile_words; ++offset) {
        const uint64_t left = left_tile[local_row][offset];
        const uint64_t right = right_tile[offset][local_column];
        const uint64_t intersection = left & right;
        // Exclusion rejects the whole set. It must not subtract excluded bits.
        if ((intersection & excluded_tile[offset]) != 0) {
          eligible = false;
          break;
        }
        if (min_chunks == 1) {
          common |= intersection;
          left_difference |= left ^ intersection;
          right_difference |= right ^ intersection;
        } else {
          count += __popcll(intersection);
        }
      }
    }
    // Ineligible threads must still participate in both block barriers.
    __syncthreads();
  }

  if (eligible) {
    eligible = min_chunks == 1
      ? common != 0 && left_difference != 0 && right_difference != 0
      : count >= min_chunks && count < lengths[row] && count < lengths[column];
  }
  if (eligible) {
    const uint32_t slot = atomicAdd(output_count, 1u);
    // A launch contains at most kBatchPairs threads, so every accepted pair
    // fits. The explicit guard also protects the FFI output on a logic error.
    if (slot < kBatchPairs) {
      output[slot] = CandidatePair{static_cast<uint32_t>(row), static_cast<uint32_t>(column)};
    }
  }
}

}  // namespace

extern "C" int rspack_cuda_probe(NativeError* error) noexcept {
  int count = 0;
  CUDA_TRY(cudaGetDeviceCount(&count));
  if (count == 0) return cuda_failure(error, "CUDA device discovery", cudaErrorNoDevice);
  int device = 0;
  CUDA_TRY(cudaGetDevice(&device));
  return 0;
}

extern "C" int rspack_cuda_round_new(
  const uint64_t* words, const uint32_t* lengths, uint32_t row_count,
  uint32_t word_count, uint32_t first, uint32_t min_chunks,
  const uint64_t* excluded, void** output, NativeError* error) noexcept {
  *output = nullptr;
  const auto allocation_start = std::chrono::steady_clock::now();
  if (first > row_count || min_chunks == 0) {
    return failure(error, -1, "Invalid CUDA round dimensions");
  }
  // Check multiplication before constructing allocation or copy sizes.
  const uint64_t total_words = static_cast<uint64_t>(row_count) * word_count;
  if (total_words > std::numeric_limits<size_t>::max() / sizeof(uint64_t)) {
    return failure(error, -1, "CUDA bitmap allocation size overflows size_t");
  }
  if (rspack_cuda_probe(error) != 0) return -1;
  std::unique_ptr<Round> round(new (std::nothrow) Round);
  if (!round) return failure(error, -1, "Could not allocate CUDA round state");
  CUDA_TRY(cudaGetDevice(&round->device));
  CUDA_TRY(cudaSetDevice(round->device));
  round->row_count = row_count;
  round->word_count = word_count;
  round->min_chunks = min_chunks;
  round->row_begin = std::max(first, 1u);

  if (word_count == 0 || round->row_begin >= row_count) {
    round->row_begin = row_count;
    round->statistics.allocation_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - allocation_start).count();
    *output = round.release();
    return 0;
  }

  const size_t word_bytes = static_cast<size_t>(total_words) * sizeof(uint64_t);
  const size_t length_bytes = static_cast<size_t>(row_count) * sizeof(uint32_t);
  const size_t excluded_bytes = excluded == nullptr ? 0
    : static_cast<size_t>(word_count) * sizeof(uint64_t);
  const size_t pair_bytes = kBatchPairs * sizeof(CandidatePair);
  if (word_bytes > std::numeric_limits<uint64_t>::max() - length_bytes
      - excluded_bytes - pair_bytes - sizeof(uint32_t)) {
    return failure(error, -1, "CUDA allocation accounting overflows u64");
  }

  CUDA_TRY(cudaStreamCreateWithFlags(&round->stream, cudaStreamNonBlocking));
  CUDA_TRY(cudaEventCreate(&round->start));
  CUDA_TRY(cudaEventCreate(&round->stop));
  CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&round->words), word_bytes));
  CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&round->lengths), length_bytes));
  if (excluded_bytes != 0) {
    CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&round->excluded), excluded_bytes));
  }
  CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&round->pairs), pair_bytes));
  CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&round->count), sizeof(uint32_t)));
  round->statistics.device_bytes = word_bytes + length_bytes + excluded_bytes
    + pair_bytes + sizeof(uint32_t);
  round->statistics.allocation_ms = std::chrono::duration<double, std::milli>(
    std::chrono::steady_clock::now() - allocation_start).count();

  CUDA_TRY(cudaEventRecord(round->start, round->stream));
  CUDA_TRY(cudaMemcpyAsync(round->words, words, word_bytes, cudaMemcpyHostToDevice,
                           round->stream));
  CUDA_TRY(cudaMemcpyAsync(round->lengths, lengths, length_bytes, cudaMemcpyHostToDevice,
                           round->stream));
  if (excluded_bytes != 0) {
    CUDA_TRY(cudaMemcpyAsync(round->excluded, excluded, excluded_bytes,
                             cudaMemcpyHostToDevice, round->stream));
  }
  if (finish_interval(*round, round->statistics.upload_ms, error) != 0) return -1;
  round->statistics.upload_bytes = word_bytes + length_bytes + excluded_bytes;
  *output = round.release();
  return 0;
}

extern "C" int rspack_cuda_round_next(
  void* filter, CandidatePair* pairs, size_t capacity, uint32_t* count,
  int* done, NativeError* error) noexcept {
  auto& round = *static_cast<Round*>(filter);
  *count = 0;
  *done = 0;
  if (capacity < kBatchPairs) {
    return failure(error, -1, "CUDA candidate output buffer is too small");
  }
  if (round.row_begin >= round.row_count) {
    *done = 1;
    return 0;
  }
  CUDA_TRY(cudaSetDevice(round.device));
  const uint32_t rows = std::min(kBatchRows, round.row_count - round.row_begin);
  const uint32_t row_end = round.row_begin + rows;
  // The last row is the only one that can pair with the largest column. No
  // later column can satisfy j < i for any row in this launch panel.
  const uint32_t columns = std::min(kBatchRows, row_end - 1 - round.column_begin);
  const uint32_t column_end = round.column_begin + columns;
  const dim3 threads(kPairTile, kPairTile);
  const dim3 blocks((columns + kPairTile - 1) / kPairTile,
                    (rows + kPairTile - 1) / kPairTile);

  CUDA_TRY(cudaMemsetAsync(round.count, 0, sizeof(uint32_t), round.stream));
  CUDA_TRY(cudaEventRecord(round.start, round.stream));
  filter_pairs<<<blocks, threads, 0, round.stream>>>(
    round.words, round.lengths, round.excluded, round.word_count, round.min_chunks,
    round.row_begin, row_end, round.column_begin, column_end, round.pairs, round.count);
  CUDA_TRY(cudaGetLastError());
  if (finish_interval(round, round.statistics.kernel_ms, error) != 0) return -1;
  ++round.statistics.launches;

  CUDA_TRY(cudaEventRecord(round.start, round.stream));
  CUDA_TRY(cudaMemcpyAsync(count, round.count, sizeof(uint32_t), cudaMemcpyDeviceToHost,
                           round.stream));
  if (finish_interval(round, round.statistics.download_ms, error) != 0) return -1;
  round.statistics.download_bytes += sizeof(uint32_t);
  if (*count > kBatchPairs || *count > capacity) {
    return failure(error, -1, "CUDA candidate count exceeded its bounded buffer");
  }
  if (*count != 0) {
    const size_t bytes = static_cast<size_t>(*count) * sizeof(CandidatePair);
    CUDA_TRY(cudaEventRecord(round.start, round.stream));
    CUDA_TRY(cudaMemcpyAsync(pairs, round.pairs, bytes, cudaMemcpyDeviceToHost, round.stream));
    if (finish_interval(round, round.statistics.download_ms, error) != 0) return -1;
    round.statistics.download_bytes += bytes;
  }
  round.statistics.accepted_pairs += *count;
  for (uint32_t row = round.row_begin; row < row_end; ++row) {
    const uint32_t end = std::min(column_end, row);
    if (end > round.column_begin) {
      round.statistics.examined_pairs += end - round.column_begin;
    }
  }

  round.column_begin = column_end;
  if (round.column_begin >= row_end - 1) {
    round.row_begin = row_end;
    round.column_begin = 0;
  }
  return 0;
}

extern "C" void rspack_cuda_round_statistics(
  const void* filter, FilterStatistics* statistics) noexcept {
  *statistics = static_cast<const Round*>(filter)->statistics;
}

extern "C" void rspack_cuda_round_free(void* filter) noexcept {
  delete static_cast<Round*>(filter);
}

#undef CUDA_TRY
