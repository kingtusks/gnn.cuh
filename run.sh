#!/bin/sh

set -xe

nvcc --compiler-options -Wall,-Wextra gnn.cu -o gnn.o -lm && ./gnn.o
