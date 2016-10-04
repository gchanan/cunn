#include "THCUNN.h"
#include "THCHalf.h"
#include "THCHalfAutoNumerics.cuh"
#include "SharedMem.cuh"

template <typename T, typename AccumulatorT>
struct MaxFloat
{
  __device__ __forceinline__ AccumulatorT operator()(AccumulatorT max, T v) const
  {
    return fmaxType(max, v);
  }
};

template<typename T, typename AccumulatorT>
struct SumFloat
{
  __device__ __forceinline__ AccumulatorT operator()(AccumulatorT sum, T v) const
  {
    return sum + v;
  }
};

template<typename T, typename AccumulatorT>
struct SumExpFloat
{
  __device__ __forceinline__ SumExpFloat(T v)
    : max_k(v)
  {}

  __device__ __forceinline__ AccumulatorT operator()(AccumulatorT sum, T v) const
  {
    return sum + THCNumerics<T>::exp(v - max_k);
  }

  const T max_k;
};

template<typename AccumulatorT>
struct NoFinal
{
  __device__ __forceinline__ AccumulatorT operator()(AccumulatorT v) const
  {
    return v;
  }
};

template<typename AccumulatorT>
struct LSMFinal
{
  __device__ __forceinline__ LSMFinal(AccumulatorT m)
    : max_k(m)
  {}

  __device__ __forceinline__ AccumulatorT operator()(AccumulatorT v) const
  {
    return max_k + THCNumerics<AccumulatorT>::exp(v);
  }

  const AccumulatorT max_k;
};

template <template<typename, typename> class Reduction, template<typename> class Finalize, typename AccumulatorT>
__device__ __forceinline__ AccumulatorT
blockReduce(AccumulatorT* smem, AccumulatorT val,
            const Reduction<AccumulatorT, AccumulatorT>& r,
            AccumulatorT defaultVal,
            const Finalize<AccumulatorT>& f)
{
  // To avoid RaW races from chaining blockReduce calls together, we
  // need a sync here
  __syncthreads();

  smem[threadIdx.x] = val;

  __syncthreads();

  AccumulatorT warpVal = defaultVal;

  // First warp will perform per-warp reductions for the remaining warps
  if ((threadIdx.x / 32) == 0) // only threads in warp1 go into this (if)
  {
    int lane = threadIdx.x % 32; // from 0 to 31

    // if less than 1024 threads per block, then only activate the relevant lanes
    if (lane < blockDim.x / 32)
    {
#pragma unroll
      for (int i = 0; i < 32; ++i)
      {
        warpVal = r(warpVal, smem[lane * 32 + i]);
      }

      smem[lane] = warpVal;
    }
  }

  __syncthreads();

  // First thread will perform a reduction of the above per-warp reductions
  AccumulatorT blockVal = defaultVal;

  if (threadIdx.x == 0)
  {
    for (int i = 0; i < blockDim.x / 32; ++i)
    {
      blockVal = r(blockVal, smem[i]);
    }

    smem[0] = f(blockVal);
  }

  // Sync and broadcast
  __syncthreads();
  return smem[0];
}

template <template<typename, typename> class Reduction, typename AccumulatorT>
__device__ __forceinline__ AccumulatorT
blockReduce(AccumulatorT* smem, AccumulatorT val,
            const Reduction<AccumulatorT, AccumulatorT>& r,
            AccumulatorT defaultVal)
{
  return blockReduce<Reduction, NoFinal, AccumulatorT>(smem, val, r, defaultVal, NoFinal<AccumulatorT>());
}

template <template<typename, typename> class Reduction, int ILP, typename T, typename AccumulatorT>
__device__ __forceinline__ AccumulatorT
ilpReduce(T* data,
          int size,
          const Reduction<T, AccumulatorT>& r,
          AccumulatorT defaultVal)
{
  AccumulatorT threadVal = defaultVal;
  int offset = threadIdx.x;

  int last = size % (ILP * blockDim.x);

  // Body (unroll by ILP times)
  for (; offset < size - last; offset += blockDim.x * ILP)
  {
    T tmp[ILP];

#pragma unroll
    for (int j = 0; j < ILP; ++j)
    {
      tmp[j] = data[offset + j * blockDim.x];
    }

#pragma unroll
    for (int j = 0; j < ILP; ++j)
    {
      threadVal = r(threadVal, tmp[j]);
    }
  }

  // Epilogue
  for (; offset < size; offset += blockDim.x)
  {
    threadVal = r(threadVal, data[offset]);
  }

  return threadVal;
}

template <int ILP, typename T, typename AccumulatorT>
__global__ void
cunn_LogSoftMax_updateOutput_kernel(T *output, T *input, int classes)
{
  SharedMem<AccumulatorT> smem;
  AccumulatorT *buffer = smem.getPointer();
  // forward pointers to batch[blockIdx.x]
  // each block handles a sample in the mini-batch
  input += blockIdx.x * classes;
  output += blockIdx.x * classes;

  // find the max of the batch
  AccumulatorT threadMax = ilpReduce<MaxFloat, ILP, T, AccumulatorT>(
      input, classes, MaxFloat<T, AccumulatorT>(), -THCNumerics<AccumulatorT>::max());
  // find the max over all batches
  AccumulatorT max_k = blockReduce<MaxFloat, AccumulatorT>(
      buffer, threadMax, MaxFloat<AccumulatorT, AccumulatorT>(), -THCNumerics<AccumulatorT>::max());
  T max_k_non_accum = ScalarConvert<AccumulatorT, T>::to(max_k);

  AccumulatorT threadExp = ilpReduce<SumExpFloat, ILP, T, AccumulatorT>(
      input, classes, SumExpFloat<T, AccumulatorT>(max_k_non_accum), 0.0);
  T logsum_k = ScalarConvert<AccumulatorT, T>::to(
      blockReduce<SumFloat, LSMFinal, AccumulatorT>(
          buffer, threadExp, SumFloat<AccumulatorT, AccumulatorT>(), 0.0, LSMFinal<AccumulatorT>(max_k)));

  // Output LSM (hand ILP)
  int offset = threadIdx.x;

  int last = classes % (ILP * blockDim.x);
  for (; offset < classes - last; offset += blockDim.x * ILP)
  {
    T tmp[ILP];

#pragma unroll
    for (int j = 0; j < ILP; ++j) {
      tmp[j] = input[offset + j * blockDim.x];
    }

#pragma unroll
    for (int j = 0; j < ILP; ++j)
    {
      output[offset + j * blockDim.x] = tmp[j] - logsum_k;
    }
  }

  for (; offset < classes; offset += blockDim.x)
  {
    output[offset] = input[offset] - logsum_k;
  }
}

template <int ILP, typename T, typename AccumulatorT>
__global__ void
cunn_LogSoftMax_updateGradInput_kernel(T *gradInput,
                                       T *output,
                                       T *gradOutput,
                                       int classes)
{
  SharedMem<AccumulatorT> smem;
  AccumulatorT *buffer = smem.getPointer();
  gradInput += blockIdx.x * classes;
  output += blockIdx.x * classes;
  gradOutput += blockIdx.x * classes;

  AccumulatorT threadSum = ilpReduce<SumFloat, 4, T, AccumulatorT>(
      gradOutput, classes, SumFloat<T, AccumulatorT>(), 0.0);
  T sum_k = ScalarConvert<AccumulatorT, T>::to(
      blockReduce<SumFloat, AccumulatorT>(
          buffer, threadSum, SumFloat<AccumulatorT, AccumulatorT>(), 0.0));

  // Update gradInput (hand ILP)
  int offset = threadIdx.x;
  int last = classes % (ILP * blockDim.x);
  for (; offset < classes - last; offset += blockDim.x * ILP)
  {
    T tmpGradOutput[ILP];
    T tmpOutput[ILP];

#pragma unroll
    for (int j = 0; j < ILP; ++j)
    {
      tmpGradOutput[j] = gradOutput[offset + j * blockDim.x];
      tmpOutput[j] = output[offset + j * blockDim.x];
    }

#pragma unroll
    for (int j = 0; j < ILP; ++j)
    {
      gradInput[offset + j * blockDim.x] =
        tmpGradOutput[j] - fastExpIfAvail(tmpOutput[j]) * sum_k;
    }
  }

  for (; offset < classes; offset += blockDim.x)
  {
    gradInput[offset] =
      gradOutput[offset] - fastExpIfAvail(output[offset]) * sum_k;
  }
}

#include "generic/LogSoftMax.cu"
#include "THCGenerateFloatTypes.h"
