#include "nodes.ispc.h"
#include <nanobind/nanobind.h>

namespace nb = nanobind;

NB_MODULE(tvb_kernels, m) {
  nb::class_<ispc::connectivity>(m, "Connectivity")
  .def(nb::init<>());
  m.def("cx_all_nop", &ispc::cx_all_nop);
}
