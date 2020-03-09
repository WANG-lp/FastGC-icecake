#pragma once

#include <pybind11/stl.h>
#include <iostream>
#include <string>
#include <vector>
#include "../../include/icecake.hpp"
#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/util/copy_with_stride.h"
namespace py = pybind11;

namespace icecake {

::dali::DALIDataType DLToDALIType(const DLDataType &dl_type);

template <typename Backend>
void CopyDlTensor(void *out_data, DLManagedTensor *dlm_tensor, cudaStream_t stream) {
    auto &dl_tensor = dlm_tensor->dl_tensor;
    auto item_size = dl_tensor.dtype.bits / 8;
    if (dl_tensor.strides) {
        std::vector<::dali::Index> strides(dl_tensor.ndim);
        for (::dali::Index i = 0; i < dl_tensor.ndim; ++i) strides[i] = dl_tensor.strides[i] * item_size;

        ::dali::CopyWithStride<Backend>(out_data, dl_tensor.data, strides.data(), dl_tensor.shape, dl_tensor.ndim,
                                        item_size, stream);

    } else {
        ::dali::CopyWithStride<Backend>(out_data, dl_tensor.data, nullptr, dl_tensor.shape, dl_tensor.ndim, item_size,
                                        stream);
    }
}
template <typename Backend>
class DaliIcecake : public ::dali::Operator<Backend> {
   public:
    inline explicit DaliIcecake(const ::dali::OpSpec &spec) : ::dali::Operator<Backend>(spec) {}

    virtual inline ~DaliIcecake() = default;

    DaliIcecake(const DaliIcecake &) = delete;
    DaliIcecake &operator=(const DaliIcecake &) = delete;
    DaliIcecake(DaliIcecake &&) = delete;
    DaliIcecake &operator=(DaliIcecake &&) = delete;

   protected:
    bool CanInferOutputs() const override { return false; }

    bool SetupImpl(std::vector<::dali::OutputDesc> &output_desc, const ::dali::workspace_t<Backend> &ws) override {
        return false;
    }

    void RunImpl(::dali::workspace_t<Backend> &ws) override;
};

}  // namespace icecake
