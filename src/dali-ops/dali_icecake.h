#pragma once

#include <pybind11/stl.h>
#include <iostream>
#include <string>
#include <vector>
#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/util/copy_with_stride.h"
namespace py = pybind11;

namespace jpegdec {

template <typename Backend>
class DaliIcecake : public ::dali::Operator<Backend> {
   public:
    inline explicit DaliIcecake(const ::dali::OpSpec &spec) : ::dali::Operator<Backend>(spec) {
    }

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
