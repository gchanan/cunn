#include "THCUNN.h"
#include "THCHalf.h"
#include "THCHalfAutoNumerics.cuh"

#include <cusparse.h>
#include <thrust/device_vector.h>

static cusparseHandle_t cusparse_handle = 0;

static void init_cusparse() {
  if (cusparse_handle == 0) {
    cusparseStatus_t status = cusparseCreate(&cusparse_handle);
    if (status != CUSPARSE_STATUS_SUCCESS) {
      THError("CUSPARSE Library initialization failed");
    }
  }
}
/*
static bool checkInput(THCudaTensor* t)
{
  return t->nDimension == 2 && t->size[1] == 3;
}

static bool checkSize2D(THCudaTensor* t, long size0, long size1)
{
  return t->nDimension == 2 && t->size[0] == size0 && t->size[1] == size1;
}

static bool checkSize1D(THCudaTensor* t, long size0)
{
  return t->nDimension == 1 && t->size[0] == size0;
}*/



#include "generic/SparseLinear.cu"
#include "THCGenerateFloatTypes.h"
