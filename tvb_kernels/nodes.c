#include "nodes.ispc.h"

#define uniform

void cx_all_j(const struct connectivity *c, int t, int j) {

  uniform float *const buf = c->buf + j * c->horizon;
  uniform int th = t + c->horizon;
#pragma omp simd
  for (int l = c->indptr[j]; l < c->indptr[j + 1]; l++) {
    int i = c->indices[l];
    float w = c->weights[l];
    int d = c->idelays[l];
    int p1 = (th - d) & c->horizon_minus_1;
    int p2 = (th - d + 1) & c->horizon_minus_1;
    c->cx1[i] += w * buf[p1];
    c->cx2[i] += w * buf[p2];
  }
}

void cx_all(const struct connectivity *c, int32_t t) {
#pragma omp simd
  for (int i = 0; i < c->num_node; i++)
    c->cx1[i] = c->cx2[i] = 0.0f;

  for (uniform int j = 0; j < c->num_node; j++)
    cx_all_j(c, t, j);
}

void cx_all2(const struct connectivity *c, int32_t t) {

  uniform int th = t + c->horizon;
#pragma omp simd
  for (int i = 0; i < c->num_node; i++) {
    float cx1 = 0.f, cx2 = 0.f;
    for (int l = c->indptr[i]; l < c->indptr[i + 1]; l++) {
      int j = c->indices[l];
      float w = c->weights[l];
      int d = c->idelays[l];
      int p1 = (th - d) & c->horizon_minus_1;
      int p2 = (th - d + 1) & c->horizon_minus_1;
      cx1 += w * c->buf[j * c->horizon + p1];
      cx2 += w * c->buf[j * c->horizon + p2];
    }
    c->cx1[i] = cx1;
    c->cx2[i] = cx2;
  }
}
void cx_all3(const struct connectivity *c, int32_t t) {

  uniform int th = t + c->horizon;
#pragma omp simd
  for (int i = 0; i < c->num_node; i++) {
    float cx1 = 0.f, cx2 = 0.f;
    for (int l = c->indptr[i]; l < c->indptr[i + 1]; l++) {
      int j = c->indices[l];
      float w = c->weights[l];
      int d = c->idelays[l];
      int p1 = (th - d) & c->horizon_minus_1;
      int p2 = (th - d + 1) & c->horizon_minus_1;
      cx1 += w * c->buf[j + p1 * c->num_node];
      cx2 += w * c->buf[j + p2 * c->num_node];
    }
    c->cx1[i] = cx1;
    c->cx2[i] = cx2;
  }
}

void cx_all_nop(const struct connectivity *c, int32_t t) {
  (void)c;
  (void)t;

  return;
}
