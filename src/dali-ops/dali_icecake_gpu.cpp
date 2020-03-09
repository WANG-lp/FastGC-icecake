#include <cuda_runtime_api.h>
#include "dali_icecake.h"

namespace icecake {

template <>
void DaliIcecake<::dali::GPUBackend>::RunImpl(::dali::workspace_t<::dali::GPUBackend> &ws) {
    auto &output = ws.OutputRef<::dali::GPUBackend>(0);
    GPUCache *gc = GC;
    // gc->load_dltensor_from_file("/tmp/dog_1.tensor");

    auto dlm_tensor = gc->get_dltensor_from_device("dog/dog_1.jpg", 0);
    output.set_type(::dali::TypeTable::GetTypeInfo(DLToDALIType(dlm_tensor->dl_tensor.dtype)));
    ::dali::TensorListShape<> list_shape{};
    list_shape.resize(batch_size_, dlm_tensor->dl_tensor.ndim);
    for (int i = 0; i < batch_size_; i++) {
        list_shape.set_tensor_shape(i, ::dali::make_span(dlm_tensor->dl_tensor.shape, dlm_tensor->dl_tensor.ndim));
    }
    output.Resize(list_shape);

    for (int i = 0; i < batch_size_; i++) {
        // CopyDlTensor<::dali::GPUBackend>(output.raw_mutable_tensor(i), dlm_tensor, ws.stream());
    }
}

}  // namespace icecake

DALI_REGISTER_OPERATOR(DaliIcecake, ::icecake::DaliIcecake<::dali::GPUBackend>, ::dali::GPU);