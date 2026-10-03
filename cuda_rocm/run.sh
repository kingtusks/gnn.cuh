#!/bin/sh

set -xe

nvcc --compiler-options -Wall,-Wextra -rdc=true -O3 gnn.cu -o gnn.o -lm && ./gnn.o
