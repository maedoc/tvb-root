#include "nodes.ispc.h"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;
using namespace nb::literals;

void conn_init_from_arrays(
    ispc::connectivity *t,
    nb::ndarray<float, nb::shape<-1>> weights,
    nb::ndarray<int32_t, nb::shape<-1>> indices,
    nb::ndarray<int32_t, nb::shape<-1>> indptr,
    nb::ndarray<int32_t, nb::shape<-1>> idelays,
    nb::ndarray<float, nb::shape<-1, -1, -1>, nb::c_contig> buf,
    nb::ndarray<float, nb::shape<-1, -1>, nb::c_contig> cx
    )
{
  new (t) ispc::connectivity {0};
  t->num_node = buf.shape(1);
  t->num_nonzero = weights.shape(0);
  t->num_cvar = buf.shape(0);
  t->horizon = buf.shape(2);
  t->horizon_minus_1 = buf.shape(2) - 1;
  if (!( (t->horizon & t->horizon_minus_1) == 0))
    throw nb::value_error("horizon (buf.shape[2]) must be power of 2");
  t->weights = weights.data();
  t->indices = indices.data();
  t->indptr = indptr.data();
  t->idelays = idelays.data();
  t->buf = buf.data();
  t->cx1 = cx.data();
  t->cx2 = cx.data() + t->num_node;
}

NB_MODULE(tvb_kernels, m) {
  nb::class_<ispc::connectivity>(m, "Connectivity")
    .def("__init__", &conn_init_from_arrays, "weights"_a, "indices"_a, "indptr"_a, "idelays"_a, "buf"_a, "cx"_a);

  m.def("cx_all_nop", &ispc::cx_all_nop);
  m.def("cx_all", &ispc::cx_all);
  m.def("cx_all_cpp", [](ispc::connectivity &c, int t) {
    for (int i=0; i<c.num_node; i++)
      c.cx1[i] = c.cx2[i] = 0;
    for (int j=0; j<c.num_node; j++)
    {
      float *buf = c.buf + j*c.horizon;
      int th = t + c.horizon;
      #pragma omp simd
      for (int l=c.indptr[j]; l<c.indptr[j+1]; l++)
      {
        int i = c.indices[l];
        float w = c.weights[l];
        int d = c.idelays[l];
        int p1 = (th - d) & c.horizon_minus_1;
        int p2 = (th - d + 1) & c.horizon_minus_1;
        c.cx1[i] += w * buf[p1];
        c.cx2[i] += w * buf[p2];
      }
    }     
  });
}
