#include "dali_icecake.h"
#include <pybind11/stl.h>
#include <cstddef>
#include <iostream>
#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/util/copy_with_stride.h"

namespace py = pybind11;

// DALI_REGISTER_OPERATOR(DaliIcecake, jpegdec::DaliIcecake, dali::CPU);
DALI_REGISTER_OPERATOR(DaliIcecake, jpegdec::DaliIcecakeMixed, Mixed);
DALI_SCHEMA(DaliIcecake).DocStr("Decoding Images with ICECAKE library.").NumInput(1).NumOutput(1).NoPrune();