#include "dali_icecake.h"
#include <iostream>
#include "../../include/icecake.hpp"
#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/util/copy_with_stride.h"
namespace icecake {
::dali::DALIDataType DLToDALIType(const DLDataType &dl_type) {
    DALI_ENFORCE(dl_type.lanes == 1, "DALI Tensors do no not support types with the number of lanes other than 1");
    switch (dl_type.code) {
        case kDLUInt: {
            switch (dl_type.bits) {
                case 8:
                    return ::dali::DALI_UINT8;
                case 16:
                    return ::dali::DALI_UINT16;
                case 32:
                    return ::dali::DALI_UINT32;
                case 64:
                    return ::dali::DALI_UINT64;
            }
            break;
        }
        case kDLInt: {
            switch (dl_type.bits) {
                case 8:
                    return ::dali::DALI_INT8;
                case 16:
                    return ::dali::DALI_INT16;
                case 32:
                    return ::dali::DALI_INT32;
                case 64:
                    return ::dali::DALI_INT64;
            }
            break;
        }
        case kDLFloat: {
            switch (dl_type.bits) {
                case 16:
                    return ::dali::DALI_FLOAT16;
                case 32:
                    return ::dali::DALI_FLOAT;
                case 64:
                    return ::dali::DALI_FLOAT64;
            }
            break;
        }
    }
    DALI_FAIL("Could not convert DLPack tensor of unsupported type ");
}

template <>
void DaliIcecake<::dali::CPUBackend>::RunImpl(::dali::workspace_t<::dali::CPUBackend> &ws) {
    auto &output = ws.OutputRef<::dali::CPUBackend>(0);
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
    auto &thread_pool = ws.GetThreadPool();
    for (int i = 0; i < batch_size_; i++) {
        thread_pool.DoWorkWithID(
            [&, i](int) { CopyDlTensor<::dali::CPUBackend>(output[i].raw_mutable_data(), dltensors[i], 0); });
    }
    thread_pool.WaitForWork();
#pragma omp parallel for
    for (size_t i = 0; i < dltensors_names.size(); i++) {
        if (dltensors[i]->deleter)
            dltensors[i]->deleter(dltensors[i]);
    }
}

}  // namespace icecake

DALI_REGISTER_OPERATOR(DaliIcecake, ::icecake::DaliIcecake<::dali::CPUBackend>, ::dali::CPU);

DALI_SCHEMA(DaliIcecake)
    .AddArg("GPUCacheObjAddr", R"code(GPUCache Python Object)code", ::dali::DALIDataType::DALI_INT64)
    .DocStr("Make a copy of the input tensor")
    .NumInput(0)
    .NumOutput(1)
    .NoPrune();