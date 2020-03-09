#include <cuda_runtime_api.h>
#include "dali_icecake.h"

namespace icecake {

template <>
void DaliIcecake<::dali::GPUBackend>::RunImpl(::dali::DeviceWorkspace &ws) {
    const auto &input = ws.Input<::dali::GPUBackend>(0);
    auto &output = ws.Output<::dali::GPUBackend>(0);
    CUDA_CALL(cudaMemcpyAsync(output.raw_mutable_data(), input.raw_data(), input.nbytes(), cudaMemcpyDeviceToDevice,
                              ws.stream()));
}

}  // namespace other_ns

DALI_REGISTER_OPERATOR(DaliIcecake, ::icecake::DaliIcecake<::dali::GPUBackend>, ::dali::GPU);