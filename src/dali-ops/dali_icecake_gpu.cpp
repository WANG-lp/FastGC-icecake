#include <cuda_runtime_api.h>
#include "dali_icecake.h"

namespace icecake {

template <>
void DaliIcecake<::dali::GPUBackend>::RunImpl(::dali::workspace_t<::dali::GPUBackend> &ws) {
    auto &output = ws.OutputRef<::dali::GPUBackend>(0);
    GPUCache *gc = GC;
    vector<std::unique_ptr<DLManagedTensor>> dltensors;
    bool hasNext = gc->next_batch(batch_size_, dltensors, true);
    DALI_ENFORCE(hasNext, "Error while geting dltensors from GPUCache");

    output.set_type(::dali::TypeTable::GetTypeInfo(DLToDALIType(dltensors[0]->dl_tensor.dtype)));
    ::dali::TensorListShape<> list_shape{};
    list_shape.resize(batch_size_, dltensors[0]->dl_tensor.ndim);
    for (int i = 0; i < batch_size_; i++) {
        list_shape.set_tensor_shape(i, ::dali::make_span(dltensors[i]->dl_tensor.shape, dltensors[i]->dl_tensor.ndim));
    }
    output.Resize(list_shape);

    for (int i = 0; i < batch_size_; i++) {
        CopyDlTensor<::dali::GPUBackend>(output.raw_mutable_tensor(i), dltensors[i].get(), ws.stream());
    }
}

}  // namespace icecake

DALI_REGISTER_OPERATOR(DaliIcecake, ::icecake::DaliIcecake<::dali::GPUBackend>, ::dali::GPU);