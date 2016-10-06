local ffi = require 'ffi'
local THNN = require 'nn.THNN'

local THCUNN = {}

-- load libTHCUNN
THCUNN.C = ffi.load(package.searchpath('libTHCUNN', package.cpath))

local THCState_ptr = ffi.typeof('THCState*')

function THCUNN.getState()
   return THCState_ptr(cutorch.getState());
end

local THCUNN_h = require 'cunn.THCUNN_h'
-- strip all lines starting with #
-- to remove preprocessor directives originally present
-- in THNN.h
THCUNN_h = THCUNN_h:gsub("\n#[^\n]*", "")
THCUNN_h = THCUNN_h:gsub("^#[^\n]*\n", "")

local THCUNN_generic_h = require 'cunn.THCUNN_generic_h'
-- strip all lines starting with #
-- to remove preprocessor directives originally present
-- in THNN.h
THCUNN_generic_h = THCUNN_generic_h:gsub("\n#[^\n]*", "")
THCUNN_generic_h = THCUNN_generic_h:gsub("^#[^\n]*\n", "")

ffi.cdef("half THC_float2half(float a);")

local preprocessed = string.gsub(THCUNN_h, 'TH_API ', '')
local preprocessed_generic = string.gsub(THCUNN_generic_h, 'TH_API void THNN_%(([%a%d_]+)%)', 'void THNN_TYPE%1')

local replacements =
{
   {
      ['THTensor'] = 'THCudaTensor',
      ['THIndexTensor'] = 'THCudaLongTensor',
      ['THIndex_t'] = 'long',
      ['THInteger_t'] = 'float'
   }
}

local cct2lt = {
   ['THCudaFloatTensor'] = 'torch.CudaTensor',
   ['THCudaDoubleTensor'] = 'torch.CudaDoubleTensor',
}

local replacements_generic =
{
  {
    ['THCTensor'] = 'THCudaTensor',
    ['THIndexTensor'] = 'THCudaLongTensor',
    ['TYPE'] = 'Cuda',
    ['real'] = 'float'
  },
  {
    ['THCTensor'] = 'THCudaDoubleTensor',
    ['THIndexTensor'] = 'THCudaLongTensor',
    ['TYPE'] = 'CudaDouble',
    ['real'] = 'double',
   }
}

if cutorch.hasHalf then
  cct2lt['THCudaHalfTensor'] = 'torch.CudaHalfTensor'
  local half_replacement = {
    ['THCTensor'] = 'THCudaHalfTensor',
    ['THIndexTensor'] = 'THCudaLongTensor',
    ['TYPE'] = 'CudaHalf',
    ['real'] = 'half'
  }
  table.insert(replacements_generic, half_replacement)
end

for i=1,#replacements do
   local r = replacements[i]
   local s = preprocessed
   for k,v in pairs(r) do
      s = string.gsub(s, k, v)
   end
   ffi.cdef(s)
end

for i=1,#replacements_generic do
    local r = replacements_generic[i]
    local s = preprocessed_generic
    for k,v in pairs(r) do
        s = string.gsub(s, k, v)
    end
    ffi.cdef(s)
end

local function extract_function_names(s)
   local t = {}
   for n in string.gmatch(s, 'TH_API void THNN_Cuda([%a%d_]+)') do
      t[#t+1] = n
   end
   return t
end

local function extract_function_names_generic(s)
   local t = {}
   for n in string.gmatch(s, 'TH_API void THNN_%(([%a%d_]+)%)') do
       t[#t+1] = n
   end
   return t
end

local function find_positions(s, p)
   local begin = 0
   local positions = {}
   while true do
      local start, stop = string.find(s, p, begin)
      if (start == nil) then break end
      positions[#positions+1] = start
      begin = stop + 1
   end
   return positions
end

local function extract_function_names_and_real_args(s)
   local t = {}
   for n in string.gmatch(s, 'TH_API ([^;]+)') do
      local func_name = string.match(n, 'void THNN_%(([%a%d_]+)%)')
      local param_positions = find_positions(n, ',')
      local positions = {}
      for x,y in ipairs(find_positions(n, "real")) do
          for cn,cp in ipairs(param_positions) do
              if cp > y then
                positions[#positions+1] = cn
                break
              end
          end
      end
   t[func_name] = positions
   end
   return t
end

local real_args = extract_function_names_and_real_args(THCUNN_generic_h)

-- build function table
local function_names = extract_function_names(THCUNN_h)
local function_names_generic = extract_function_names_generic(THCUNN_generic_h)

-- combine function names for CudaTensor
for k,v in pairs(real_args) do
  function_names[#function_names+1] = k
end

THNN.kernels['torch.CudaTensor'] = THNN.bind(THCUNN.C, function_names, 'Cuda', THCUNN.getState)
torch.getmetatable('torch.CudaTensor').THNN = THNN.kernels['torch.CudaTensor']

THNN.kernels['torch.CudaDoubleTensor'] = THNN.bind(THCUNN.C, function_names_generic, 'CudaDouble', THCUNN.getState)
torch.getmetatable('torch.CudaDoubleTensor').THNN = THNN.kernels['torch.CudaDoubleTensor']

-- in order to call 'half' functions from lua, convert real arguments from
-- to half since there is no other defined conversion
local transform_reals_to_half = function(t, func_name, real_args)
    for k,v in ipairs(real_args[func_name]) do
        -- first argument (THCState) is added implicitly by bind
        t[v-1] = ffi.C.THC_float2half(t[v-1])
    end
    return t
end

local raw_half_functions = THNN.bind(THCUNN.C, function_names_generic, 'CudaHalf', THCUNN.getState)
for k,v in pairs(raw_half_functions) do
    raw_half_functions[k] = function(...) v(unpack(transform_reals_to_half({...}, k, real_args)))
end
end
THNN.kernels['torch.CudaHalfTensor'] = raw_half_functions
torch.getmetatable('torch.CudaHalfTensor').THNN = THNN.kernels['torch.CudaHalfTensor']

return THCUNN
