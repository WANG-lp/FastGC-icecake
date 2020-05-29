#include <cuda_runtime_api.h>
#include "dali_icecake.h"

namespace jpegdec {

template <>
void DaliIcecake<::dali::GPUBackend>::RunImpl(::dali::workspace_t<::dali::GPUBackend> &ws) {
    auto &output = ws.OutputRef<::dali::GPUBackend>(0);
}

}  // namespace jpegdec

DALI_REGISTER_OPERATOR(DaliIcecake, ::jpegdec::DaliIcecake<::dali::GPUBackend>, ::dali::GPU);