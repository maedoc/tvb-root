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

void cx_all_arrays(
  nb::ndarray<float, nb::shape<-1>> weights_,
  nb::ndarray<int32_t, nb::shape<-1>> indices_,
  nb::ndarray<int32_t, nb::shape<-1>> indptr_,
  nb::ndarray<int32_t, nb::shape<-1>> idelays_,
  nb::ndarray<float, nb::shape<-1, -1, -1>, nb::c_contig> buf_,
  nb::ndarray<float, nb::shape<-1, -1>, nb::c_contig> cx_,
  const int32_t t
) {
  int num_node = buf_.shape(1);
  int num_nonzero = weights_.shape(0);
  int num_cvar = buf_.shape(0);
  int horizon = buf_.shape(2);
  int horizon_minus_1 = buf_.shape(2) - 1;

  auto weights = weights_.view<float, nb::ndim<1> >();
  auto indices = indices_.view<int32_t, nb::ndim<1> >();
  auto indptr = indptr_.view<int32_t, nb::ndim<1> >();
  auto idelays = idelays_.view<int32_t, nb::ndim<1> >();
  auto buf = buf_.view<float, nb::ndim<3> >();
  auto cx = cx_.view<float, nb::ndim<2>, nb::shape<2, -1> >();

  for (int i=0; i<num_node; i++)
    cx(0, i) = cx(1, i) = 0;

  for (int j=0; j<num_node; j++)
  {
    int th = t + horizon;
    #pragma omp simd
    for (int l=indptr(j); l<indptr(j+1); l++)
    {
      int i = indices(l);
      float w = weights(l);
      int d = idelays(l);
      int p1 = (th - d) & horizon_minus_1;
      int p2 = (th - d + 1) & horizon_minus_1;
      cx(0, i) += w * buf(0, j, p1);
      cx(1, i) += w * buf(0, j, p2);
    }
  }
}

NB_MODULE(tvb_kernels, m) {
  nb::class_<ispc::connectivity>(m, "Connectivity")
    .def("__init__", &conn_init_from_arrays, "weights"_a, "indices"_a, "indptr"_a, "idelays"_a, "buf"_a, "cx"_a);

  m.def("cx_all_nop", &ispc::cx_all_nop);
  m.def("cx_all", &ispc::cx_all);
  m.def("cx_all_arrays", &cx_all_arrays,
        "weights"_a, "indices"_a, "indptr"_a, "idelays"_a, "buf"_a, "cx"_a, "t"_a);

  m.def("cx_all_cpp", [](ispc::connectivity &c, int t)
  {
    for (int i=0; i<c.num_node; i++)
      c.cx1[i] = c.cx2[i] = 0;

    for (int j=0; j<c.num_node; j++)
    {
      float *buf = c.buf + j*c.horizon;
      int th = t + c.horizon;

      // #pragma omp simd
      for (int l=c.indptr[j]; l<c.indptr[j+1]; l++)
      {
        int i = c.indices[l];
        float w = c.weights[l];
        int d = c.idelays[l];
        int p1 = (th -  d) & c.horizon_minus_1;
        int p2 = (th -  d + 1) & c.horizon_minus_1;
        c.cx1[i] += w * buf[p1];
        c.cx2[i] += w * buf[p2];
      }
    }
  });

}
