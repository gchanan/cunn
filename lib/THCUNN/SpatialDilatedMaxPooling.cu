#include "THCUNN.h"
#include "THCHalf.h"
#include "THCHalfAutoNumerics.cuh"
#include "THCAtomics.cuh"
#include "common.h"

// kernels borrowed from Caffe
template <typename Dtype, typename AccType>
__global__ void MaxPoolForward(const int nthreads, const Dtype* bottom_data,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const int pad_h, const int pad_w,
    const int dilation_h, const int dilation_w, Dtype* top_data,
    long* top_mask) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    int hend = min(hstart + (kernel_h - 1) * dilation_h + 1, height);
    int wend = min(wstart + (kernel_w - 1) * dilation_w + 1, width);
    while(hstart < 0)
      hstart += dilation_h;
    while(wstart < 0)
      wstart += dilation_w;
    float maxval = -FLT_MAX;
    int maxidx = -1;
    bottom_data += (n * channels + c) * height * width;
    for (int h = hstart; h < hend; h += dilation_h) {
      for (int w = wstart; w < wend; w += dilation_w) {
        Dtype my_bottom_data = bottom_data[h * width + w];
        //printf("my bottom data is: %f \n", ScalarConvert<Dtype, AccType>::to(my_bottom_data));
        if (ScalarConvert<Dtype, AccType>::to(bottom_data[h * width + w]) > maxval) {
          maxidx = h * width + w;
          maxval = ScalarConvert<Dtype, AccType>::to(bottom_data[maxidx]);
        }
      }
    }
    top_data[index] = ScalarConvert<AccType, Dtype>::to(maxval);
    top_mask[index] = maxidx + TH_INDEX_BASE;
  }
}


template <typename Dtype, typename AccType>
__global__ void MaxPoolBackward(const int nthreads, const Dtype* top_diff,
    const long* top_mask, const int num, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
    const int dilation_h, const int dilation_w,
    Dtype* bottom_diff) {
  //if ()
  if ((threadIdx.x == 133 || threadIdx.x == 134) && blockIdx.x == 120) {
    printf("Loop stuff: blockId: %i blockDim: %i threadId: %i gridDi: %i\n", blockIdx.x, blockDim.x, threadIdx.x, gridDim.x);
  }
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;
    int phstart =
        (h + pad_h < ((kernel_h - 1) * dilation_h + 1)) ? 0 : (h + pad_h - ((kernel_h - 1) * dilation_h + 1)) / stride_h + 1;
    int phend = min((h + pad_h) / stride_h + 1, pooled_height);
    int pwstart =
        (w + pad_w < ((kernel_w - 1) * dilation_w + 1)) ? 0 : (w + pad_w - ((kernel_w - 1) * dilation_w + 1)) / stride_w + 1;
    int pwend = min((w + pad_w) / stride_w + 1, pooled_width);

    AccType gradient = AccType(0);
    int offset = (n * channels + c) * pooled_height * pooled_width;
    top_diff += offset;
    top_mask += offset;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
	       if (top_mask[ph * pooled_width + pw] - TH_INDEX_BASE == h * width + w) {
            AccType oldGradient = gradient;
            int asdf = ph * pooled_width + pw;
            AccType myTopDiff = ScalarConvert<Dtype, AccType>::to(top_diff[ph * pooled_width + pw]);
	          gradient += myTopDiff;
            if (index == 61573 || index == 61574) {
              printf("adding %f to gradient: %f now: %f for index: %i topdiffindex: %i threadId: %i blockId: %i, blockDim: %i\n", myTopDiff, oldGradient, gradient, index, asdf, threadIdx.x, blockIdx.x, blockDim.x);
            }
	       }
      }
    }
    int reduced = index - 209920;
    int altIndex = index;
    //printf("WRITING to gradient2: %f %i %i %i %i %i %i\n", gradient, phstart, phend, pwstart, pwend, index, reduced);
    if (index == 61573 || index == 61574) {
      //printf("OMG WE FOUND IT!\n");
      printf("WRITING TO gradient: %f for index: %i\n", gradient, index);
    }
    bottom_diff[index] = ScalarConvert<AccType, Dtype>::to(gradient);
    //#ifdef THC_REAL_IS_HALF
    //bottom_diff[index].x = ScalarConvert<int, Dtype>::to(index);//ScalarConvert<AccType, Dtype>::to(gradient);
    //#else
    //bottom_diff[index] = ScalarConvert<AccType, Dtype>::to(gradient);
    //#endif
  }
}

#include "generic/SpatialDilatedMaxPooling.cu"
#include "THCGenerateFloatTypes.h"
