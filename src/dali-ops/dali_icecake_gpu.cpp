#include <cuda_runtime_api.h>
#include "dali_icecake.h"

namespace icecake {

template <>
void DaliIcecake<::dali::GPUBackend>::RunImpl(::dali::workspace_t<::dali::GPUBackend> &ws) {
    auto &output = ws.OutputRef<::dali::GPUBackend>(0);
    GPUCache *gc = GC;
    vector<string> dltensors_names;
    bool hasNext = gc->next_batch(batch_size_, dltensors_names, true);
    DALI_ENFORCE(hasNext, "Error while geting dltensors from GPUCache");
    vector<DLManagedTensor *> dltensors;
    dltensors.resize(dltensors_names.size());

#pragma omp parallel for
    for (size_t i = 0; i < dltensors_names.size(); i++) {
        dltensors[i] = gc->get_dltensor_from_device(dltensors_names[i], 0);
    }

    output.set_type(::dali::TypeTable::GetTypeInfo(DLToDALIType(dltensors[0]->dl_tensor.dtype)));
    ::dali::TensorListShape<> list_shape{};
    list_shape.resize(batch_size_, dltensors[0]->dl_tensor.ndim);
    for (int i = 0; i < batch_size_; i++) {
        list_shape.set_tensor_shape(i, ::dali::make_span(dltensors[i]->dl_tensor.shape, dltensors[i]->dl_tensor.ndim));
    }
    output.Resize(list_shape);

    for (int i = 0; i < batch_size_; i++) {
        CopyDlTensor<::dali::GPUBackend>(output.raw_mutable_tensor(i), dltensors[i], ws.stream());
    }
#pragma omp parallel for
    for (size_t i = 0; i < dltensors_names.size(); i++) {
        if (dltensors[i]->deleter)
            dltensors[i]->deleter(dltensors[i]);
    }
}

}  // namespace icecake

DALI_REGISTER_OPERATOR(DaliIcecake, ::icecake::DaliIcecake<::dali::GPUBackend>, ::dali::GPU);