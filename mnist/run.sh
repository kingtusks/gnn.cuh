#!/bin/sh

set -xe

g++ -Wall -Wextra -std=c++11 mnist.cpp -o mnist.o && ./mnist.o
