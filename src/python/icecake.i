%module pyicecake
%begin %{
#define SWIG_PYTHON_STRICT_BYTE_CHAR
%}

%{ 
#include "../include/icecake.hpp"
%}

%include "std_vector.i"
%include "std_string.i"
%include "stdint.i"

%template(StringVector) std::vector<std::string>;
%template(IntVector) std::vector<int>;
%template(CharVector) std::vector<char>;


%typemap(in) int32_t
{
    $1 = PyInt_AsInt($input);
}


%include "../include/icecake.hpp"