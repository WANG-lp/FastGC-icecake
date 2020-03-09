#ifndef EXAMPLE_DUMMY_H_
#define EXAMPLE_DUMMY_H_

#include <vector>

#include "dali/pipeline/operator/operator.h"

namespace icecake {

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
    bool CanInferOutputs() const override { return true; }

    bool SetupImpl(std::vector<::dali::OutputDesc> &output_desc, const ::dali::workspace_t<Backend> &ws) override {
        const auto &input = ws.template InputRef<Backend>(0);
        output_desc.resize(1);
        output_desc[0] = {input.shape(), input.type()};
        return true;
    }

    void RunImpl(::dali::Workspace<Backend> &ws) override;
};

} 

#endif  // EXAMPLE_DUMMY_H_