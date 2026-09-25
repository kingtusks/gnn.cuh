#!/bin/sh

set -xe

nvcc --compiler-options -Wall,-Wextra -rdc=true gnn.cu -o gnn.o -lm && ./gnn.o
