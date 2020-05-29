#include "dali_icecake.h"
#include <iostream>
#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/util/copy_with_stride.h"
namespace jpegdec {

template <>
void DaliIcecake<::dali::CPUBackend>::RunImpl(::dali::workspace_t<::dali::CPUBackend> &ws) {
    auto &output = ws.OutputRef<::dali::CPUBackend>(0);
}

}  // namespace jpegdec

DALI_REGISTER_OPERATOR(DaliIcecake, ::jpegdec::DaliIcecake<::dali::CPUBackend>, ::dali::CPU);

DALI_SCHEMA(DaliIcecake)
    .AddArg("GPUCacheObjAddr", R"code(GPUCache Python Object)code", ::dali::DALIDataType::DALI_INT64)
    .DocStr("Make a copy of the input tensor")
    .NumInput(0)
    .NumOutput(1)
    .NoPrune();