#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/SpatialDilatedMaxPooling.cu"
#else

#include "../common.h"

void THNN_(SpatialDilatedMaxPooling_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *output,
           THCudaLongTensor *indices,
           int kW, int kH,
           int dW, int dH,
           int padW, int padH,
           int dilationW, int dilationH,
           bool ceil_mode)
{

  THCUNN_assertSameGPU_generic(state, 3, input, output, indices);
  THArgCheck(input->nDimension == 3 || input->nDimension == 4, 2, "3D or 4D (batch) tensor expected");

  long nInputCols, nInputRows, nInputPlane, batchSize;
  long nOutputCols, nOutputRows;

  if (input->nDimension == 3) {
    nInputCols = input->size[2];
    nInputRows = input->size[1];
    nInputPlane = input->size[0];
    batchSize = 1;
  }
  else
  {
    nInputCols = input->size[3];
    nInputRows = input->size[2];
    nInputPlane = input->size[1];
    batchSize = input->size[0];
  }

  THArgCheck(nInputCols >= kW - padW && nInputRows >= kH - padH, 2, "input image smaller than kernel size");
  THArgCheck(kW/2 >= padW && kH/2 >= padH, 2, "pad should be smaller than half of kernel size");

  if(ceil_mode) {
    nOutputCols = ceil(float(nInputCols - (dilationW * (kW - 1) + 1) + 2*padW) / float(dW)) + 1;
    nOutputRows = ceil(float(nInputRows - (dilationH * (kH - 1) + 1) + 2*padH) / float(dH)) + 1;
  }
  else {
    nOutputCols = floor(float(nInputCols - (dilationW * (kW - 1) + 1) + 2*padW) / float(dW)) + 1;
    nOutputRows = floor(float(nInputRows - (dilationH * (kH - 1) + 1) + 2*padH) / float(dH)) + 1;
  }

if (nOutputCols < 1 || nOutputRows < 1)
    THError("Given input size: (%dx%dx%d). Calculated output size: (%dx%dx%d). Output size is too small",
            nInputPlane,nInputRows,nInputCols,nInputPlane,nOutputRows,nOutputCols);

if (padW || padH)
  {
    // ensure that the last pooling starts inside the image
    if ((nOutputRows - 1)*dH >= nInputRows + padH)
      --nOutputRows;
    if ((nOutputCols  - 1)*dW >= nInputCols  + padW)
      --nOutputCols;
  }

  input = THCTensor_(newContiguous)(state, input);
  real* input_data = THCTensor_(data)(state, input);

  THCTensor_(resize4d)(state, output, batchSize, nInputPlane, nOutputRows, nOutputCols);
  THCUNN_resizeAs_indices(state, indices, output);

  long* indices_data = THCudaLongTensor_data(state, indices);
  real* output_data = THCTensor_(data)(state, output);

  int count = THCTensor_(nElement)(state, output);

  MaxPoolForward<real, accreal> <<< GET_BLOCKS(count), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state) >>>
      (count, input_data,
      batchSize, nInputPlane, nInputRows, nInputCols, nOutputRows, nOutputCols,
      kH, kW, dH, dW, padH, padW, dilationH, dilationW, output_data, indices_data);
  THCudaCheck(cudaGetLastError());

  if(input->nDimension == 3)
    THCTensor_(resize3d)(state, output, nInputPlane, nOutputRows, nOutputCols);

  THCTensor_(free)(state, input);
}

void THNN_(SpatialDilatedMaxPooling_updateGradInput)(
           THCState *state,
           THCTensor *input,
           THCTensor *gradOutput,
           THCTensor *gradInput,
           THCudaLongTensor *indices,
           int kW, int kH,
           int dW, int dH,
           int padW, int padH,
           int dilationW, int dilationH,
           bool ceil_mode)
{
  THCUNN_assertSameGPU_generic(state, 4, input, gradOutput, indices, gradInput);
//print("rescuda", rescuda:sub(12,12,37,37,34,35)))
  //real entry1 = THCTensor_(get3d)(state, gradOutput, 12,10,12);
  //real entry2 = THCTensor_(get3d)(state, gradOutput, 12,37,35);(12,12,10,10,12,12))
  //printf("entry (12,10,12) is: %f\n", ScalarConvert<real, accreal>::to(entry1));
  //real entry1 = THCTensor_(get3d)(state, gradOutput, 11,9,11);
  //printf("entry (12,10,12) is: %f\n", ScalarConvert<real, accreal>::to(entry1));
  //printf("entry (12,37,35) is: %f\n", ScalarConvert<real, accreal>::to(entry2));
  input = THCTensor_(newContiguous)(state, input);
  gradOutput = THCTensor_(newContiguous)(state, gradOutput);
  //entry1 = THCTensor_(get3d)(state, gradOutput, 11,9,11);
  //printf("entry (12,10,12) is: %f\n", ScalarConvert<real, accreal>::to(entry1));

  long nInputCols, nInputRows, nInputPlane, batchSize;
  long nOutputCols, nOutputRows;

  if (input->nDimension == 3) {
    nInputCols = input->size[2];
    nInputRows = input->size[1];
    nInputPlane = input->size[0];
    batchSize = 1;
  }
  else
  {
    nInputCols = input->size[3];
    nInputRows = input->size[2];
    nInputPlane = input->size[1];
    batchSize = input->size[0];
  }

  if(ceil_mode) {
     nOutputCols = ceil(float(nInputCols - (dilationW * (kW - 1) + 1) + 2*padW) / float(dW)) + 1;
     nOutputRows = ceil(float(nInputRows - (dilationH * (kH - 1) + 1) + 2*padH) / float(dH)) + 1;
   }
   else {
     nOutputCols = floor(float(nInputCols - (dilationW * (kW - 1) + 1) + 2*padW) / float(dW)) + 1;
     nOutputRows = floor(float(nInputRows - (dilationH * (kH - 1) + 1) + 2*padH) / float(dH)) + 1;
   }

  if (nOutputCols < 1 || nOutputRows < 1)
    THError("Given input size: (%dx%dx%d). Calculated output size: (%dx%dx%d). Output size is too small",
            nInputPlane,nInputRows,nInputCols,nInputPlane,nOutputRows,nOutputCols);

  gradOutput = THCTensor_(newContiguous)(state, gradOutput);
  THCTensor_(resizeAs)(state, gradInput, input);

  int count = THCTensor_(nElement)(state, input);
  printf("COUNTS: %i %i", count, GET_BLOCKS(count));

  MaxPoolBackward<real, accreal> <<< GET_BLOCKS(count), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state) >>>
  //MaxPoolBackward<real, accreal> <<< 1, 1, 0, THCState_getCurrentStream(state) >>>
      (count,
      THCTensor_(data)(state, gradOutput),
      THCudaLongTensor_data(state, indices),
      batchSize, nInputPlane, nInputRows, nInputCols, nOutputRows, nOutputCols,
      kH, kW, dH, dW, padH, padW, dilationH, dilationW,
      THCTensor_(data)(state, gradInput));
  THCudaCheck(cudaGetLastError());

  THCTensor_(free)(state, gradOutput);

  // clean
  THCTensor_(free)(state, input);
  THCTensor_(free)(state, gradOutput);
}

#endif
