set -e
g++ main.cc -I oneDNN/include -I oneDNN/build/include/ -I oneDNN/examples/ -L oneDNN/build/src -ldnnl
LD_LIBRARY_PATH=oneDNN/build/src/ ./a.out
