#include <stdint.h>

typedef struct params params;

struct params {
  const int count;
  const float *values;
};

// connectivity model with csr format sparse connections & delay buffer
typedef struct connectivity connectivity;

struct connectivity {
  const int num_node;
  const int num_nonzero;
  const int num_cvar;
  // horizon must be power of two
  const int horizon;
  const int horizon_minus_1;
  const float *weights; // (num_nonzero,)
  const int *indices; // (num_nonzero,)
  const int *indptr; // (num_nodes+1,)
  const int *idelays; // (num_nonzero,)
  float *buf; // delay buffer (num_cvar, num_nodes, horizon)
  float *cx1;
  float *cx2;
};

typedef struct sim sim;
struct sim {
  // keep invariant stuff at the top, per sim stuff below
  const int rng_seed;
  const int num_node;
  const int num_svar;
  const int num_time;
  const int num_params;
  const int num_spatial_params;
  const float dt;
  const int oversample; // TODO "oversample" for stability,
  const int num_skip; // skip per output sample
  float *z_scale; // (num_svar), sigma*sqrt(dt)

  // parameters
  const params global_params;
  const params spatial_params;

  float *state_trace; // (num_time//num_skip, num_svar, num_nodes)
  float *states; // full states (num_svar, num_nodes)

  const connectivity conn;
};

typedef const sim the_sim;
typedef const connectivity the_conn;

void cx_all(the_conn *c, int32_t t);
void cx_all2(the_conn *c, int32_t t);
void cx_all3(the_conn *c, int32_t t);
void cx_all_nop(the_conn *c, int32_t t);
