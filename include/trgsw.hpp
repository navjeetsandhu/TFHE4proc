#pragma once
#include <memory>
#include <array>
#include <cstdint>
#include <iostream>
#include "mulfft.hpp"
#include "params.hpp"
#include "trlwe.hpp"
#include "decomposition.hpp"
namespace TFHEpp {

template <class P>
TRGSWFFT<P> ApplyFFT2trgsw(const TRGSW<P> &trgsw)
{
    std::cout << "a";
    alignas(64) TRGSWFFT<P> trgswfft;
    for (int i = 0; i < (P::k + 1) * P::l; i++)
        for (int j = 0; j < (P::k + 1); j++)
            TwistIFFT<P>(trgswfft[i][j], trgsw[i][j]);
    return trgswfft;
}

template <class P, int batch>
TRGSWFFTn<P, batch> ApplyFFT2trgswbatch(const TRGSWn<P, batch> &trgsw)
{
    alignas(64) TRGSWFFTn<P, batch> trgswfft;
    for (int i = 0; i < (P::k + 1) * P::l; i++)
        for (int j = 0; j < (P::k + 1); j++)
            TwistIFFTbatch<P, batch>(trgswfft[i][j], trgsw[i][j]);
    return trgswfft;
}

template <class P>
TRGSW<P> trgswSymEncrypt(const Polynomial<P> &p, const double alpha,
                         const Key<P> &key)
{
    constexpr std::array<typename P::T, P::l> h = hgen<P>();

    TRGSW<P> trgsw;
    for (TRLWE<P> &trlwe : trgsw) trlwe = trlweSymEncryptZero<P>(alpha, key);
    for (int i = 0; i < P::l; i++) {
        for (int k = 0; k < P::k + 1; k++) {
            for (int j = 0; j < P::n; j++) {
                trgsw[i + k * P::l][k][j] +=
                    static_cast<typename P::T>(p[j]) * h[i];
            }
        }
    }
    return trgsw;
}


template <class P, int batch>
TRGSWn<P, batch> trgswSymEncryptbatch(const Polynomialn<P, batch> &p, const double alpha,
                         const Key<P> &key)
{
    constexpr std::array<typename P::T, P::l> h = hgen<P>();
    //std::cout << " trgswSymEncryptbatch: " << std::endl;
    std::unique_ptr<TRGSWn<P, batch>> trgsWnPtr = std::make_unique<TRGSWn<P, batch>>();
    for (TRLWEn<P, batch> &trlwe : (*trgsWnPtr)) trlwe = trlweSymEncryptZerobatch<P, batch>(alpha, key);
    for (int i = 0; i < P::l; i++)
        for (int k = 0; k < P::k + 1; k++)
            for (int b = 0; b < batch; b++) {
                //std::cout << b << " ";
                for (int j = 0; j < P::n; j++)
                    (*trgsWnPtr)[i + k * P::l][k][b][j] +=
                        static_cast<typename P::T>(p[b][j]) * h[i];
            }

    return (*trgsWnPtr);
}

template <class P>
TRGSW<P> trgswSymEncrypt(const Polynomial<P> &p, const uint eta,
                         const Key<P> &key)
{
    constexpr std::array<typename P::T, P::l> h = hgen<P>();

    TRGSW<P> trgsw;
    for (TRLWE<P> &trlwe : trgsw) trlwe = trlweSymEncryptZero<P>(eta, key);
    for (int i = 0; i < P::l; i++) {
        for (int k = 0; k < P::k + 1; k++) {
            for (int j = 0; j < P::n; j++) {
                trgsw[i + k * P::l][k][j] +=
                    static_cast<typename P::T>(p[j]) * h[i];
            }
        }
    }
    return trgsw;
}

template <class P, int batch>
TRGSWn<P, batch> trgswSymEncryptbatch(const Polynomialn<P, batch> &p, const uint eta,
                         const Key<P> &key)
{
    constexpr std::array<typename P::T, P::l> h = hgen<P>();

    TRGSWn<P, batch> trgsw;
    for (TRLWEn<P, batch> &trlwe : trgsw) trlwe = trlweSymEncryptZerobatch<P, batch>(eta, key);

    for (int i = 0; i < P::l; i++)
        for (int k = 0; k < P::k + 1; k++)
            for (int b = 0; b < batch; b++)
                for (int j = 0; j < P::n; j++)
                    trgsw[i + k * P::l][k][b][j] += static_cast<typename P::T>(p[b][j]) * h[i];

    return trgsw;
}

template <class P>
TRGSW<P> trgswSymEncrypt(const Polynomial<P> &p, const Key<P> &key)
{
    if constexpr (P::errordist == ErrorDistribution::ModularGaussian)
        return trgswSymEncrypt<P>(p, P::alpha, key);
    else
        return trgswSymEncrypt<P>(p, P::eta, key);
}

template <class P, int batch>
TRGSWn<P, batch> trgswSymEncryptbatch(const Polynomialn<P, batch> &p, const Key<P> &key)
{
    if constexpr (P::errordist == ErrorDistribution::ModularGaussian)
        return trgswSymEncryptbatch<P, batch>(p, P::alpha, key);
    else
        return trgswSymEncryptbatch<P, batch>(p, P::eta, key);
}

template <class P>
TRGSWFFT<P> trgswfftSymEncrypt(const Polynomial<P> &p, const double alpha,
                               const Key<P> &key)
{
    std::cout << "j";
    TRGSW<P> trgsw = trgswSymEncrypt<P>(p, alpha, key);
    return ApplyFFT2trgsw<P>(trgsw);
}

template <class P, int batch>
TRGSWFFTn<P, batch> trgswfftSymEncryptbatch(const Polynomialn<P, batch> &p, const double alpha,
                               const Key<P> &key)
{
    TRGSWn<P, batch> trgsw = trgswSymEncryptbatch<P, batch>(p, alpha, key);
    return ApplyFFT2trgswbatch<P, batch>(trgsw);
}

template <class P>
TRGSWFFT<P> trgswfftSymEncrypt(const Polynomial<P> &p, const uint eta,
                               const Key<P> &key)
{
    std::cout << "J";
    TRGSW<P> trgsw = trgswSymEncrypt<P>(p, eta, key);
    return ApplyFFT2trgsw<P>(trgsw);
}

template <class P, int batch>
TRGSWFFT<P> trgswfftSymEncryptbatch(const Polynomialn<P, batch> &p, const uint eta,
                               const Key<P> &key)
{
    TRGSWn<P, batch> trgsw = trgswSymEncryptbatch<P, batch>(p, eta, key);
    return ApplyFFT2trgswbatch<P, batch>(trgsw);
}

template <class P>
TRGSWFFT<P> trgswfftSymEncrypt(const Polynomial<P> &p, const Key<P> &key)
{
    if constexpr (P::errordist == ErrorDistribution::ModularGaussian)
        return trgswfftSymEncrypt<P>(p, P::alpha, key);
    else
        return trgswfftSymEncrypt<P>(p, P::eta, key);
}

template <class P, int batch>
TRGSWFFTn<P, batch> trgswfftSymEncryptbatch(const Polynomialn<P, batch> &p, const Key<P> &key)
{
    if constexpr (P::errordist == ErrorDistribution::ModularGaussian)
        return trgswfftSymEncryptbatch<P, batch>(p, P::alpha, key);
    else
        return trgswfftSymEncryptbatch<P, batch>(p, P::eta, key);
}

}  // namespace TFHEpp