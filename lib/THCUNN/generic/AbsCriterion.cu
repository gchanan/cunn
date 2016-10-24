#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/AbsCriterion.cu"
#else

void THNN_(AbsCriterion_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *target,
           THCTensor *output,
           bool sizeAverage)
{
  THCUNN_assertSameGPU_generic(state, 2, input, target);

  long size = THCTensor_(nElement)(state, input);

  input = THCTensor_(newContiguous)(state, input);
  target = THCTensor_(newContiguous)(state, target);

  thrust::device_ptr<real> input_data(THCTensor_(data)(state, input));
  thrust::device_ptr<real> target_data(THCTensor_(data)(state, target));
  real sum = thrust::inner_product(input_data, input_data+size, target_data, ScalarConvert<int, real>::to(0), thrust::plus<real>(), abs_functor<real>());

  if (sizeAverage)
    sum /= size;

  THCTensor_(free)(state, input);
  THCTensor_(free)(state, target);

  THCTensor_(set1d)(state, output, 0, sum);
}

void THNN_(AbsCriterion_updateGradInput)(
           THCState *state,
           THCTensor *input,
           THCTensor *target,
           THCTensor *gradInput,
           bool sizeAverage)
{
  THCUNN_assertSameGPU_generic(state, 3, input, target, gradInput);

  long size = THCTensor_(nElement)(state, input);
  real norm = ScalarConvert<double, real>::to(sizeAverage ? 1./size : 1.);

  input = THCTensor_(newContiguous)(state, input);
  target = THCTensor_(newContiguous)(state, target);

  THCTensor_(resizeAs)(state, gradInput, input);

  thrust::device_ptr<real> input_data(THCTensor_(data)(state, input));
  thrust::device_ptr<real> target_data(THCTensor_(data)(state, target));
  thrust::device_ptr<real> gradInput_data(THCTensor_(data)(state, gradInput));

  thrust::transform(input_data, input_data+size, target_data, gradInput_data, abs_updateGradInput_functor<real>(norm));

  THCTensor_(free)(state, input);
  THCTensor_(free)(state, target);
}

#endif
