# TFHEfft
TFHEfft is full Scracthed pure C++ Ver. of TFHE. 

# Building

### build using FPGA 
./fftBuild.sh


# Supported Compiler
GCC9.1 later are primarily supported compilers.

# Third party libraries
Codes under thirdparties directory contain third-party libraries, Randen, Cereal, and SPQLIOS. See the corresponding directory to check the licenses.

### Randen
TFHEfft uses this as a Cryptographically Secure Pseudo-Random Number Generator (CSPRNG). Original repository is [here](https://github.com/google/randen).

### Cereal
Cereal is a header-only C++11 serialization library. TFHEfft uses this to export ciphertexts and keys. Cereal is treated by the git submodule.

### FFTW3
[FFTW](https://www.fftw.org/) is one of the most famous FFT libraries.
To install FFT libraries:
$ sudo apt-get install libfftw3-dev
To check where the install put stuff (not the install would have installed additional packages in addition to just the base libfftw3 package.
$ dpkg -L libfftw3-dev


### SPQLIOS
SPQLIOS is the FFT library using AVX2 that is dedicated to the ring R\[X\]/(X^N+1) for N a power of 2.

### FPGA
TFHEfft uses FPGA by default
./fftBuild.sh