#pragma once
#include <memory>
#include "mulfft.hpp"
#include "params.hpp"
#include <iostream>

namespace TFHEpp {
template <class P>
TRLWE<P> trlweSymEncryptZero(const double alpha, const Key<P> &key)
{
    //std::cout << "g";
    constexpr auto numeric_limit = std::numeric_limits<typename P::T>::max(); // i.e. 0xFFFFFFFF
    constexpr auto dimension = P::n; // i.e. 1024
    constexpr auto k_max = P::k;  // i.e 1
    std::uniform_int_distribution<typename P::T> Torusdist(0, numeric_limit);
    TRLWE<P> c;
    for (typename P::T &i : c[k_max]) i = ModularGaussian<P>(0, alpha);
    for (int k = 0; k < k_max; k++) {
        for (typename P::T &i : c[k]) i = Torusdist(generator);
        std::array<typename P::T, dimension> partkey;
        for (int i = 0; i < dimension; i++) partkey[i] = key[k * dimension + i];
        Polynomial<P> temp;
        PolyMul<P>(temp, c[k], partkey);
        for (int i = 0; i < dimension; i++) c[k_max][i] += temp[i];
    }
    return c;
}

template <class P, int batch>
TRLWEn<P, batch> trlweSymEncryptZerobatch(const double alpha, const Key<P> &key)
{
    //std::cout << " trlweSymEncryptZerobatch ";
        constexpr auto numeric_limit = std::numeric_limits<typename P::T>::max(); // i.e. 0xFFFFFFFF
    constexpr auto dimension = P::n; // i.e. 1024
    constexpr auto k_max = P::k;  // i.e 1
    std::uniform_int_distribution<typename P::T> Torusdist(0, numeric_limit);
    std::unique_ptr<TRLWEn<P, batch>> cPtr = std::make_unique<TRLWEn<P, batch>>();

    for (int j=0;j<batch;j++)
        for (typename P::T &i : (*cPtr)[k_max][j]) i = ModularGaussian<P>(0, alpha);

    for (int k = 0; k < k_max; k++) {
        for (int j=0;j<batch;j++)
            for (typename P::T &i : (*cPtr)[k][j]) i = Torusdist(generator);

        std::array<typename P::T, dimension> partkey;
        for (int i = 0; i < dimension; i++) partkey[i] = key[k * dimension + i];

        std::unique_ptr<Polynomialn<P, batch>> tempPtr = std::make_unique<Polynomialn<P, batch>>();
        PolyMulHalfbatch<P, batch>((*tempPtr), (*cPtr)[k], partkey);

        for (int j=0;j<batch;j++)
            for (int i = 0; i < dimension; i++)
                (*cPtr)[k_max][j][i] += (*tempPtr)[j][i];

    }
    return (*cPtr);
}

template <class P>
TRLWE<P> trlweSymEncryptZero(const uint eta, const Key<P> &key)
{
    std::cout << "G";
    constexpr auto numeric_limit = std::numeric_limits<typename P::T>::max(); // i.e. 0xFFFFFFFF
    constexpr auto dimension = P::n; // i.e. 1024
    constexpr auto k_max = P::k; // i.e 1
    std::uniform_int_distribution<typename P::T> Torusdist(0, numeric_limit);

    alignas(64) TRLWE<P> c;
    for (typename P::T &i : c[k_max])
        i = (CenteredBinomial<P>(eta) << std::numeric_limits<P>::digits) / P::q;


    for (int k = 0; k < k_max; k++) {
        for (typename P::T &i : c[k]) i = Torusdist(generator);
        alignas(64) std::array<typename P::T, P::n> partkey;
        for (int i = 0; i < dimension; i++) partkey[i] = key[k * dimension + i];
        alignas(64) Polynomial<P> temp;
        PolyMul<P>(temp, c[k], partkey);
        for (int i = 0; i < dimension; i++) c[k_max][i] += temp[i];
    }
    return c;
}

template <class P, int batch>
TRLWEn<P, batch> trlweSymEncryptZerobatch(const uint eta, const Key<P> &key)
{
    //std::cout << " trlweSymEncryptZerobatch ";
    constexpr auto numeric_limit = std::numeric_limits<typename P::T>::max(); // i.e. 0xFFFFFFFF
    constexpr auto dimension = P::n; // i.e. 1024
    constexpr auto k_max = P::k; // i.e 1
    std::uniform_int_distribution<typename P::T> Torusdist(0, numeric_limit);
    alignas(64) TRLWEn<P, batch> c;

    for (int j=0;j<batch;j++)
        for (typename P::T &i : c[k_max][j])
            i = (CenteredBinomial<P>(eta) << std::numeric_limits<P>::digits) / P::q;

    for (int k = 0; k < k_max; k++) {
        for (int j=0;j<batch;j++)
            for (typename P::T &i : c[k][j]) i = Torusdist(generator);

        alignas(64) std::array<typename P::T, P::n> partkey;
        for (int i = 0; i < dimension; i++) partkey[i] = key[k * dimension + i];

        alignas(64) Polynomialn<P, batch> temp;
        for (int j=0;j<batch;j++) {
            PolyMul<P>(temp[j], c[k], partkey);
            for (int i = 0; i < dimension; i++) c[k_max][j][i] += temp[j][i];
        }
    }
    return c;
}

template <class P>
TRLWE<P> trlweSymEncryptZero(const Key<P> &key)
{
    if constexpr (P::errordist == ErrorDistribution::ModularGaussian)
        return trlweSymEncryptZero<P>(P::alpha, key);
    else
        return trlweSymEncryptZero<P>(P::eta, key);
}

template <class P, int batch>
TRLWEn<P,batch> trlweSymEncryptZerobatch(const Key<P> &key)
{
    if constexpr (P::errordist == ErrorDistribution::ModularGaussian)
        return trlweSymEncryptZerobatch<P,batch>(P::alpha, key);
    else
        return trlweSymEncryptZerobatch<P,batch>(P::eta, key);
}


template <class P>
TRLWE<P> trlweSymEncrypt(const std::array<typename P::T, P::n> &p,
                         const double alpha, const Key<P> &key)
{
    std::cout << "k";
    TRLWE<P> c = trlweSymEncryptZero<P>(alpha, key);
    for (int i = 0; i < P::n; i++) c[P::k][i] += p[i];
    return c;
}

template <class P, int batch>
TRLWEn<P, batch> trlweSymEncryptbatch(const Polynomialn<P, batch> &p,
                         const double alpha, const Key<P> &key)
{
    TRLWEn<P, batch> c = trlweSymEncryptZerobatch<P, batch>(alpha, key);
    for (int j=0;j<batch;j++)
        for (int i = 0; i < P::n; i++) c[P::k][j][i] += p[j][i];
    return c;
}


template <class P>
TRLWE<P> trlweSymEncrypt(const std::array<typename P::T, P::n> &p, const uint eta,
                         const Key<P> &key)
{
    std::cout << "K";
    TRLWE<P> c = trlweSymEncryptZero<P>(eta, key);
    for (int i = 0; i < P::n; i++) c[P::k][i] += p[i];
    return c;
}

template <class P, int batch>
TRLWEn<P, batch> trlweSymEncryptbatch(const Polynomialn<P, batch> &p, const uint eta,
                         const Key<P> &key)
{
    TRLWEn<P, batch> c = trlweSymEncryptZerobatch<P, batch>(eta, key);
    for (int j=0;j<batch;j++)
        for (int i = 0; i < P::n; i++) c[P::k][j][i] += p[j][i];
    return c;
}



template <class P>
TRLWE<P> trlweSymEncrypt(const std::array<typename P::T, P::n> &p,
                         const Key<P> &key)
{
    if constexpr (P::errordist == ErrorDistribution::ModularGaussian)
        return trlweSymEncrypt<P>(p, P::alpha, key);
    else
        return trlweSymEncrypt<P>(p, P::eta, key);
}

template <class P, int batch>
TRLWEn<P, batch> trlweSymEncryptbatch(const Polynomialn<P, batch> &p,
                         const Key<P> &key)
{
    if constexpr (P::errordist == ErrorDistribution::ModularGaussian)
        return trlweSymEncryptbatch<P, batch>(p, P::alpha, key);
    else
        return trlweSymEncryptbatch<P, batch>(p, P::eta, key);
}


template <class P>
TRLWE<P> trlweSymIntEncrypt(const std::array<typename P::T, P::n> &p,
                            const double alpha, const Key<P> &key)
{
    TRLWE<P> c = trlweSymEncryptZero<P>(alpha, key);
    for (int i = 0; i < P::n; i++)
        c[P::k][i] += static_cast<typename P::T>(P::delta * p[i]);
    return c;
}

template <class P, int batch>
TRLWE<P> trlweSymIntEncryptbatch(const Polynomialn<P, batch> &p,
                            const double alpha, const Key<P> &key)
{
    TRLWEn<P, batch> c = trlweSymEncryptZerobatch<P, batch>(alpha, key);

    for (int j=0;j<batch;j++)
        for (int i = 0; i < P::n; i++)
            c[P::k][j][i] += static_cast<typename P::T>(P::delta * p[j][i]);
    return c;
}



template <class P>
TRLWE<P> trlweSymIntEncrypt(const std::array<typename P::T, P::n> &p,
                            const uint eta, const Key<P> &key)
{
    TRLWE<P> c = trlweSymEncryptZero<P>(eta, key);
    for (int i = 0; i < P::n; i++)
        c[P::k][i] += static_cast<typename P::T>(P::delta * p[i]);
    return c;
}

template <class P, int batch>
TRLWEn<P,batch> trlweSymIntEncryptbatch(const Polynomialn<P, batch> &p,
                            const uint eta, const Key<P> &key)
{
    TRLWEn<P, batch> c = trlweSymEncryptZerobatch<P, batch>(eta, key);

    for (int j=0;j<batch;j++)
        for (int i = 0; i < P::n; i++)
            c[P::k][j][i] += static_cast<typename P::T>(P::delta * p[j][i]);
    return c;
}


template <class P>
TRLWE<P> trlweSymIntEncrypt(const std::array<typename P::T, P::n> &p,
                            const Key<P> &key)
{
    if constexpr (P::errordist == ErrorDistribution::ModularGaussian)
        return trlweSymIntEncrypt<P>(p, P::alpha, key);
    else
        return trlweSymIntEncrypt<P>(p, P::eta, key);
}


template <class P, int batch>
TRLWEn<P, batch> trlweSymIntEncryptbatch(const Polynomialn<P, batch> &p,
                            const Key<P> &key)
{
    if constexpr (P::errordist == ErrorDistribution::ModularGaussian)
        return trlweSymIntEncryptbatch<P, batch>(p, P::alpha, key);
    else
        return trlweSymIntEncryptbatch<P, batch>(p, P::eta, key);
}


template <class P>
Polynomial<P> trlwePhase(const TRLWE<P> &c, const Key<P> &key)
{
    //std::cout << " trlwePhase ";
    Polynomial<P> phase = c[P::k];
    for (int k = 0; k < P::k; k++) {
        Polynomial<P> mulres;
        std::array<typename P::T, P::n> partkey;
        for (int i = 0; i < P::n; i++) partkey[i] = key[k * P::n + i];
        PolyMul<P>(mulres, c[k], partkey);
        for (int i = 0; i < P::n; i++) phase[i] -= mulres[i];
    }
    return phase;
}

template <class P, int batch>
Polynomialn<P, batch> trlwePhasebatch(const TRLWEn<P, batch> &c, const Key<P> &key)
{
    //std::cout << " trlwePhasebatch ";
    Polynomialn<P, batch> phase = c[P::k];
    for (int k = 0; k < P::k; k++) {
        alignas(64) std::unique_ptr<Polynomialn<P, batch>> mulresPtr = std::make_unique<Polynomialn<P, batch>>();
        std::array<typename P::T, P::n> partkey;
        for (int i = 0; i < P::n; i++) partkey[i] = key[k * P::n + i];
        PolyMulHalfbatch<P, batch>(*mulresPtr, c[k], partkey);

        for (int j=0;j<batch;j++)
            for (int i = 0; i < P::n; i++) phase[j][i] -= (*mulresPtr)[j][i];

    }
    return phase;
}


template <class P>
std::array<bool, P::n> trlweSymDecrypt(const TRLWE<P> &c, const Key<P> &key)
{
    Polynomial<P> phase = trlwePhase<P>(c, key);

    std::array<bool, P::n> p;
    if constexpr (hasq<P>::value) {
        for (int i = 0; i < P::n; i++) p[i] = (phase[i] % P::q) < P::q / 2;
    }
    else
        for (int i = 0; i < P::n; i++)
            p[i] = static_cast<typename std::make_signed<typename P::T>::type>(
                       phase[i]) > 0;
    return p;
}

template <class P, int batch>
BooleanArrayn<P::n, batch> trlweSymDecryptbatch(const TRLWEn<P, batch> &c, const Key<P> &key)
{
    //std::cout << " trlweSymDecryptbatch ";
    Polynomialn<P, batch> phase = trlwePhasebatch<P, batch>(c, key);

    BooleanArrayn<P::n, batch> p;
    if constexpr (hasq<P>::value) {
        for (int j=0;j<batch;j++)
            for (int i = 0; i < P::n; i++) p[j][i] = (phase[j][i] % P::q) < P::q / 2;
    }
    else
        for (int j=0;j<batch;j++)
            for (int i = 0; i < P::n; i++)
                p[j][i] = static_cast<typename std::make_signed<typename P::T>::type>(
                       phase[j][i]) > 0;
    return p;
}


template <class P>
Polynomial<P> trlweSymIntDecrypt(const TRLWE<P> &c, const Key<P> &key)
{
    //std::cout << " trlweSymIntDecrypt ";
    Polynomial<P> phase = c[P::k];
    for (int k = 0; k < P::k; k++) {
        Polynomial<P> mulres;
        std::array<typename P::T, P::n> partkey;
        for (int i = 0; i < P::n; i++) partkey[i] = key[k * P::n + i];
        PolyMul<P>(mulres, c[k], partkey);
        for (int i = 0; i < P::n; i++) phase[i] -= mulres[i];
    }

    Polynomial<P> p;
    for (int i = 0; i < P::n; i++)
        p[i] = static_cast<typename P::T>(std::round(phase[i] / P::delta)) %
               P::plain_modulus;
    return p;
}

template <class P, int batch>
Polynomialn<P, batch> trlweSymIntDecrypt(const TRLWEn<P, batch> &c, const Key<P> &key)
{
    //std::cout << " trlweSymIntDecryptbatch ";
    Polynomialn<P, batch> phase = c[P::k];
    for (int k = 0; k < P::k; k++) {
        for (int j=0;j<batch;j++) {
            Polynomial<P> mulres;
            std::array<typename P::T, P::n> partkey;
            for (int i = 0; i < P::n; i++) partkey[i] = key[k * P::n + i];
            PolyMul<P>(mulres, c[j][k], partkey);
            for (int i = 0; i < P::n; i++) phase[j][i] -= mulres[i];
        }
    }

    Polynomialn<P, batch> p;
    for (int i = 0; i < P::n; i++)
        for (int j=0;j<batch;j++)
            p[j][i] = static_cast<typename P::T>(std::round(phase[j][i] / P::delta)) %
               P::plain_modulus;
    return p;
}


template <class P>
void SampleExtractIndex(TLWE<P> &tlwe, const TRLWE<P> &trlwe, const int index)
{
    for (int k = 0; k < P::k; k++) {
        for (int i = 0; i <= index; i++)
            tlwe[k * P::n + i] = trlwe[k][index - i];
        for (int i = index + 1; i < P::n; i++)
            tlwe[k * P::n + i] = -trlwe[k][P::n + index - i];
    }
    tlwe[P::k * P::n] = trlwe[P::k][index];
}

template <class P, int batch>
void SampleExtractIndexbatch(TLWEn<P, batch> &tlwe, const TRLWEn<P, batch> &trlwe, const int index)
{
    for (int j= 0; j< batch;j++) {
        for (int k = 0; k < P::k; k++) {
            for (int i = 0; i <= index; i++)
                tlwe[j][k * P::n + i] = trlwe[k][j][index - i];
            for (int i = index + 1; i < P::n; i++)
                tlwe[j][k * P::n + i] = -trlwe[k][j][P::n + index - i];
        }
        tlwe[j][P::k * P::n] = trlwe[P::k][j][index];
    }
}


}  // namespace TFHEpp