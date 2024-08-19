#!/bin/bash

src=$1
set -eux

h=${src}.h
obj=${src}.o
dll=${src}.so
mod=${src%".ispc"}
py=${mod}.py

ispc -g -O3 --pic ${src} -h ${h} -o ${obj}
gcc -shared ${obj} -o ${dll}
rm -f ${py}
ctypesgen -l ./${dll} ${h} > ${py}

gcc-14 -g -O3 -fopenmp -mavx2 -march=native -mtune=native -c nodes.c -o nodes.c.o
gcc-14 -shared nodes.c.o -o nodes.c.so
