#pragma once
#include "mult_fft_fpga.hpp"
#include <fft_processor_fftw.h>
#include <iostream>
#include <memory>

namespace TFHEpp {

template <class P, int batch>
inline void TwistFFTbatch(Polynomialn<P, batch> &res, const PolynomialInFDn<P, batch> &a)
{
    //std::cout << "b";
    if constexpr (std::is_same_v<P, lvl1param>)
        TwistFpgaFFTbatch(res[0].data(), a[0].data(), batch);
    else
        static_assert(false_v<typename P::T>, "Undefined TwistFFT batch!");
}

template <class P, int batch>
inline void TwistIFFTbatch(PolynomialInFDn<P, batch> &res, const Polynomialn<P, batch> &a)
{
    //std::cout << "B";
    if constexpr (std::is_same_v<P, lvl1param>)
        TwistFpgaIFFTbatch(res[0].data(), a[0].data(), batch);
    else
        static_assert(false_v<typename P::T>, "Undefined TwistIFFT batch!");
}

template <class P>
inline void TwistFFT(Polynomial<P> &res, const PolynomialInFD<P> &a)
{
    std::cout << "*";
    if constexpr (std::is_same_v<P, lvl1param>)
        TwistFpgaFFT<P::n>(res, a);
    else if constexpr (std::is_same_v<typename P::T, uint64_t>)
        fftplvl2.execute_direct_torus64(res.data(), a.data());
    else
        static_assert(false_v<typename P::T>, "Undefined TwistFFT!");
}


template <class P>
inline void TwistFFTrescale(Polynomial<P> &res, const PolynomialInFD<P> &a)
{
    std::cout << "&";
    if constexpr (std::is_same_v<P, lvl1param>)
        TwistFpgaFFTrescale<P>(res, a);
    else if constexpr (std::is_same_v<P, lvl2param>)
        fftplvl2.execute_direct_torus64_rescale(res.data(), a.data(), P::delta);
    else
        static_assert(false_v<typename P::T>, "Undefined TwistFFT!");
}

template <class P>
inline void TwistIFFT(PolynomialInFD<P> &res, const Polynomial<P> &a)
{
    std::cout << "%";
    if constexpr (std::is_same_v<P, lvl1param>)
        TwistFpgaIFFT<P::n>(res, a);
    else if constexpr (std::is_same_v<typename P::T, uint64_t>)
        fftplvl2.execute_reverse_torus64(res.data(), a.data());
    else
        static_assert(false_v<typename P::T>, "Undefined TwistIFFT!");
}
template <uint32_t N>
inline void MulInFD(std::array<double, N> &res, const std::array<double, N> &b)
{
    for (int i = 0; i < N / 2; i++) {
        double aimbim = res[i + N / 2] * b[i + N / 2];
        double arebim = res[i] * b[i + N / 2];
        res[i] = std::fma(res[i], b[i], -aimbim);
        res[i + N / 2] = std::fma(res[i + N / 2], b[i], arebim);
    }
}

template <class P, int batch>
inline void MulInFDbatch(PolynomialInFDn<P, batch> &res,
                         const PolynomialInFDn<P, batch> &b)
{
    for (int i=0; i< batch; i++)
        MulInFD<P::n>(res[i], b[i]);
}


template <uint32_t N>
inline void MulInFD(std::array<double, N> &res, const std::array<double, N> &a,
                    const std::array<double, N> &b)
{
    for (int i = 0; i < N / 2; i++) {
        res[i] = a[i] * b[i];
        res[i + N / 2] = a[i + N / 2] * b[i];
    }
    for (int i = 0; i < N / 2; i++) {
        res[i + N / 2] += a[i] * b[i + N / 2];
        res[i] -= a[i + N / 2] * b[i + N / 2];
    }
}

template <class P, int batch>
inline void MulInFDbatch(PolynomialInFDn<P, batch> &res, const PolynomialInFDn<P, batch> &a,
                         const PolynomialInFDn<P, batch> &b)
{
    for (int i=0; i< batch; i++) {
        MulInFD<P::n>(res[i], a[i], b[i]);
    }
}

template <class P, int batch>
inline void MulInFDbatch(PolynomialInFDn<P, batch> &res, const PolynomialInFDn<P, batch> &a,
                         const PolynomialInFD<P> &b)
{
    for (int i=0; i< batch; i++) {
        MulInFD<P::n>(res[i], a[i], b);
    }
}


// Be careful about memory accesses (We assume b has relatively high memory access cost)
template <uint32_t N>
inline void FMAInFD(std::array<double, N> &res, const std::array<double, N> &a,
                    const std::array<double, N> &b)
{
    for (int i = 0; i < N / 2; i++) {
        res[i] = std::fma(a[i], b[i], res[i]);
        res[i + N / 2] = std::fma(a[i + N / 2], b[i], res[i + N / 2]);
    }
    for (int i = 0; i < N / 2; i++) {
        res[i + N / 2] = std::fma(a[i], b[i + N / 2], res[i + N / 2]);
        res[i] -= a[i + N / 2] * b[i + N / 2];
    }
}


template <class P, int batch>
inline void FMAInFDbatch(PolynomialInFDn<P, batch> &res, const PolynomialInFDn<P, batch> &a,
                         const PolynomialInFDn<P, batch> &b)
{
    for (int j=0; j< batch; j++) {
        FMAInFD<P::n>(res[j], a[j], b[j]);
    }
}

template <class P, int batch>
inline void FMAInFDbatch(PolynomialInFDn<P, batch> &res, const PolynomialInFDn<P, batch> &a,
                         const PolynomialInFD<P> &b)
{
    for (int j=0; j< batch; j++) {
        FMAInFD<P::n>(res[j], a[j], b);
    }
}


template <class P>
inline void PolyMulNaive(Polynomial<P> &res, const Polynomial<P> &a,
                         const Polynomial<P> &b)
{
    for (int i = 0; i < P::n; i++) {
        typename P::T ri = 0;
        for (int j = 0; j <= i; j++)
            ri += static_cast<typename std::make_signed<typename P::T>::type>(
                      a[j]) *
                  b[i - j];
        for (int j = i + 1; j < P::n; j++)
            ri -= static_cast<typename std::make_signed<typename P::T>::type>(
                      a[j]) *
                  b[P::n + i - j];
        res[i] = ri;
    }
}


template <class P, int batch>
inline void PolyMulFFTbatch(Polynomialn<P, batch> &res, const Polynomialn<P, batch> &a,
                       const Polynomialn<P, batch> &b)
{
    //std::cout << " PolyMulFFTbatch ";
    alignas(64) PolynomialInFDn<P, batch> ffta;
    TwistIFFTbatch<P, batch>(ffta, a);
    alignas(64) PolynomialInFDn<P, batch> fftb;
    TwistIFFTbatch<P, batch>(fftb, b);
    MulInFDbatch<P, batch>(ffta, fftb);
    TwistFFTbatch<P, batch>(res, ffta);
}



template <class P>
inline void PolyMulFFT(Polynomial<P> &res, const Polynomial<P> &a,
                       const Polynomial<P> &b)
{
    std::cout << "c";
    alignas(64) PolynomialInFD<P> ffta;
    TwistIFFT<P>(ffta, a);
    alignas(64) PolynomialInFD<P> fftb;
    TwistIFFT<P>(fftb, b);
    MulInFD<P::n>(ffta, fftb);
    TwistFFT<P>(res, ffta);
}


template <class P>
inline void PolyMul(Polynomial<P> &res, const Polynomial<P> &a,
                    const Polynomial<P> &b)
{
    if constexpr (std::is_same_v<typename P::T, uint32_t>) {
        PolyMulFFT<P>(res, a, b);
    }
    else {
        PolyMulNaive<P>(res, a, b);
    }
}

template <class P, int batch>
inline void PolyMulbatch(Polynomialn<P, batch> &res, const Polynomialn<P, batch> &a,
                    const Polynomialn<P, batch> &b)
{
    if constexpr (std::is_same_v<typename P::T, uint32_t>)
        PolyMulFFTbatch<P, batch>(res, a, b);
    else
        static_assert(false_v<typename P::T>, "PolyMulbatch!");

}

template <class P, int batch>
inline void PolyMulHalfbatch(Polynomialn<P, batch> &res, const Polynomialn<P, batch> &a,
                         const Polynomial<P> &b)
{
    alignas(64) std::unique_ptr<Polynomialn<P, batch>> bPtr = std::make_unique<Polynomialn<P, batch>>();

    for (int j= 0; j< batch;j++)
        for (int i = 0; i < P::n; i++)
            (*bPtr)[j][i] = b[i];

    PolyMulbatch<P, batch>(res, a, *bPtr);
}




template <class P>
inline void PolyMulRescaleUnsigned(Polynomial<P> &res,
                                   const UnsignedPolynomial<P> &a,
                                   const UnsignedPolynomial<P> &b)
{
    std::cout << "d";
    PolynomialInFD<P> ffta, fftb;
    TwistIFFT<P>(ffta, a);
    TwistIFFT<P>(fftb, b);
    MulInFD<P::n>(ffta, ffta, fftb);
    TwistFFTrescale<P>(res, ffta);
}

}  // namespace TFHEpp