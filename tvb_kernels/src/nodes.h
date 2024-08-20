struct params {
  const int count;
  data_floats values;
};

// connectivity model with csr format sparse connections & delay buffer
struct connectivity {
  const uniform int num_node;
  const uniform int num_nonzero;
  const uniform int num_cvar;
  // horizon must be power of two
  const int horizon;
  const int horizon_minus_1;
  data_floats weights; // (num_nonzero,)
  data_ints indices; // (num_nonzero,)
  data_ints indptr; // (num_nodes+1,)
  data_ints idelays; // (num_nonzero,)
  work_floats buf; // delay buffer (num_cvar, num_nodes, horizon)
  work_floats cx1;
  work_floats cx2;
};

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
  work_floats z_scale; // (num_svar), sigma*sqrt(dt)

  // parameters
  const uniform params global_params;
  const uniform params spatial_params;

  work_floats state_trace; // (num_time//num_skip, num_svar, num_nodes)
  work_floats states; // full states (num_svar, num_nodes)

  const uniform connectivity conn;
};

typedef const uniform struct sim the_sim;
typedef const uniform struct connectivity the_conn;

