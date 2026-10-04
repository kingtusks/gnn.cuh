#!/bin/sh

set -xe

cmake --build build && ./build/gnn
