# tvb_kernels

This is a library of computational kernels for TVB.

## scope

in order of priority

- [ ] sparse delay coupling functions
- [ ] fused heun neural mass model step functions
- [ ] neural ODE
- [ ] bold / tavg monitors

## building

For now, a `make` invocation is enough, which calls `mkispc.sh` to build
the ISPC kernels, then CMake to build the nanobind wrappers.  The next
steps will convert this to a pure CMake based process.
 
## variants

### implementation

ISPC compiler provides the best performance, followed by a C/C++ compiler.
To handle cases where all compilers are not available, the pattern will
be as follows

Python class owns kernel workspace (arrays, floats, int) and has a
short, obvious NumPy based implementation.

A nanobind binding layer optionally implements calls to
- a short, obvious C++ implementation
- an (possibly multithreaded) ISPC implementation
- a CUDA implementation, if arrays are on GPU

### batch variants

It may be desirable to compute batches of a kernel.

### parameters

Parameters may be global or "spatialized" meaning different
values for every node.  Kernels should provide separate calls
for both cases.

## matlab

Given some users in matlab, it could be useful to wrap some of the algorithms
as MEX files.

## working notes

### inner loop

for the inner loop, excl monitors etc, all that's needed is
- cx_all
- step: stochastic heun on local modelo
- buf update

### scalable design

#### parameter sweeps

separating the data from the work arrays, allows defining
the data once and reusing it, which is important for large
connectivities.

the work arrays can then be allocated for each simulation to do.

#### components

enabling a multicomponent design requires some modifications.
as a useful example consider a model with the following elements

- cortical field with modified epileptor
- subcortical regions with Jansen-Rit
- corticortical excitatory connectivity
- subcortical excitattory connectivity
- cortical->subcortical inhibitory connectivity
- subcortical->cortical inhibitory connectivity
- dopamine region
- dopa->cortical connectivity

which implies some notions we already have. let's use the
word *domain* for field, regions, etc and *projection* for
the different connectivities

- a domain has one or more input *ports* which sum afferent
  values from various projections
- a domain has one or more output *ports* which expose efferent
  values read by various projections
- a single projection may transport one or more values

changes required

- proj should not own the afferent cx buffer
  rather take it as a operator argument like t
  or just have a pointer
- a given model should own the input port buffer
- projections should own the delay buffer
- delay buffers should have num_cvars as last dimension

- once step taken, push to "listening" projections

