#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/LogSoftMax.cu"
#else

#include "../common.h"

void THNN_(LogSoftMax_updateOutput)(
          THCState *state,
          THCTensor *input,
          THCTensor *output)
{
  THCUNN_assertSameGPU_generic(state, 2, input, output);

  input = THCTensor_(newContiguous)(state, input);
  THCTensor_(resizeAs)(state, output, input);

  int batchSize = 1;
  int classSize = 0;

  if (THCTensor_(nDimension)(state, input) == 1)
  {
    classSize = THCTensor_(size)(state, input, 0);
  }
  else if (THCTensor_(nDimension)(state, input) == 2)
  {
    batchSize = THCTensor_(size)(state, input, 0);
    classSize = THCTensor_(size)(state, input, 1);
  }
  else
  {
    THError("vector or matrix expected");
  }

  dim3 grid(batchSize);
  dim3 block(1024);

  cunn_LogSoftMax_updateOutput_kernel<2, real, accreal>
    <<<grid, block, block.x * sizeof(real), THCState_getCurrentStream(state)>>>(
      THCTensor_(data)(state, output),
      THCTensor_(data)(state, input),
      classSize
  );
  THCudaCheck(cudaGetLastError());
  THCTensor_(free)(state, input);
}

void THNN_(LogSoftMax_updateGradInput)(
           THCState *state,
           THCTensor *input,
           THCTensor *gradOutput,
           THCTensor *gradInput,
           THCTensor *output)
{
  THCUNN_assertSameGPU_generic(state, 3, output, gradOutput, gradInput);

  output = THCTensor_(newContiguous)(state, output);
  gradOutput = THCTensor_(newContiguous)(state, gradOutput);

  THCTensor_(resizeAs)(state, gradInput, output);

  int batchSize = 1;
  int classSize = 0;

  if (THCTensor_(nDimension)(state, gradInput) == 1)
  {
    classSize = THCTensor_(size)(state, gradInput, 0);
  }
  else if (THCTensor_(nDimension)(state, gradInput) == 2)
  {
    batchSize = THCTensor_(size)(state, gradInput, 0);
    classSize = THCTensor_(size)(state, gradInput, 1);
  }
  else
  {
    THError("vector or matrix expected");
  }

  dim3 grid(batchSize);
  dim3 block(1024);

  cunn_LogSoftMax_updateGradInput_kernel<2, real, accreal>
    <<<grid, block, block.x * sizeof(real), THCState_getCurrentStream(state)>>>(
      THCTensor_(data)(state, gradInput),
      THCTensor_(data)(state, output),
      THCTensor_(data)(state, gradOutput),
      classSize
  );
  THCudaCheck(cudaGetLastError());
  THCTensor_(free)(state, gradOutput);
  THCTensor_(free)(state, output);
}

#endif
