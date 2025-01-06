#pragma once
#include "trgsw.hpp"
#include "externalproduct.hpp"
#include <iostream>

namespace TFHEpp {
template <class P>
void CMUXFFT(TRLWE<P> &res, const TRGSWFFT<P> &cs, const TRLWE<P> &c1,
             const TRLWE<P> &c0)
{
    for (int k = 0; k < P::k + 1; k++)
        for (int i = 0; i < P::n; i++) res[k][i] = c1[k][i] - c0[k][i];
    trgswfftExternalProduct<P>(res, res, cs);
    for (int k = 0; k < P::k + 1; k++)
        for (int i = 0; i < P::n; i++) res[k][i] += c0[k][i];
}

template <class P>
TRGSWFFT<P> TRGSWFFTOneGen()
{
    //std::cout << "h";
    constexpr std::array<typename P::T, P::l> h = hgen<P>();
    TRGSW<P> trgsw = {};
    for (int i = 0; i < P::l; i++) {
        for (int k = 0; k < P::k + 1; k++) {
            trgsw[i + k * P::l][k][0] += h[i];
        }
    }
    return ApplyFFT2trgsw<P>(trgsw);
}

alignas(64) const TRGSWFFT<lvl1param> trgswonelvl1 =
    TRGSWFFTOneGen<lvl1param>();
alignas(64) const TRGSWFFT<lvl2param> trgswonelvl2 =
    TRGSWFFTOneGen<lvl2param>();

template <class bkP>
void CMUXFFTwithPolynomialMulByXaiMinusOne(
    TRLWE<typename bkP::targetP> &acc,
    const BootstrappingKeyElementFFT<bkP> &cs, const int a)
{
    if constexpr (bkP::domainP::key_value_diff == 1) {
        alignas(64) TRLWE<typename bkP::targetP> temp;
        for (int k = 0; k < bkP::targetP::k + 1; k++)
            PolynomialMulByXaiMinusOne<typename bkP::targetP>(temp[k], acc[k],
                                                              a);
        trgswfftExternalProduct<typename bkP::targetP>(temp, temp, cs[0]);
        for (int k = 0; k < bkP::targetP::k + 1; k++)
            for (int i = 0; i < bkP::targetP::n; i++) acc[k][i] += temp[k][i];
    }
    else {
        alignas(32) TRLWE<typename bkP::targetP> temp;
        int count = 0;
        for (int i = bkP::domainP::key_value_min;
             i <= bkP::domainP::key_value_max; i++) {
            if (i != 0) {
                const int mod = (a * i) % (2 * bkP::targetP::n);
                const int index = mod > 0 ? mod : mod + (2 * bkP::targetP::n);
                for (int k = 0; k < bkP::targetP::k + 1; k++)
                    PolynomialMulByXaiMinusOne<typename bkP::targetP>(
                        temp[k], acc[k], index);
                trgswfftExternalProduct<typename bkP::targetP>(temp, temp,
                                                               cs[count]);
                for (int k = 0; k < bkP::targetP::k + 1; k++)
                    for (int n = 0; n < bkP::targetP::n; n++)
                        acc[k][n] += temp[k][n];
                count++;
            }
        }
    }
}

template <class bkP, int batch>
void CMUXFFTwithPolynomialMulByXaiMinusOnebatch(
    TRLWEn<typename bkP::targetP, batch> &acc,
    const BootstrappingKeyElementFFT<bkP> &cs, const intArray<batch> &aArray)
{
    if constexpr (bkP::domainP::key_value_diff == 1) {
        alignas(64) TRLWEn<typename bkP::targetP, batch> temp;
        for (int j = 0; j < batch; j++)
            for (int k = 0; k < bkP::targetP::k + 1; k++)
                PolynomialMulByXaiMinusOne<typename bkP::targetP>(temp[k][j], acc[k][j],
                                                                  aArray[j]);
        trgswfftExternalProductbatch<typename bkP::targetP, batch>(temp, temp, cs[0]);
        for (int j = 0; j < batch; j++)
            for (int k = 0; k < bkP::targetP::k + 1; k++)
                for (int i = 0; i < bkP::targetP::n; i++) acc[k][j][i] += temp[k][j][i];
    }
    else {
        alignas(32) TRLWEn<typename bkP::targetP, batch> temp;
        int count = 0;
        for (int i = bkP::domainP::key_value_min;
             i <= bkP::domainP::key_value_max; i++) {
            if (i != 0) {

                for (int j = 0; j < batch; j++) {
                    const int mod = (aArray[j] * i) % (2 * bkP::targetP::n);
                    const int index = mod > 0 ? mod : mod + (2 * bkP::targetP::n);
                    for (int k = 0; k < bkP::targetP::k + 1; k++)
                        PolynomialMulByXaiMinusOne<typename bkP::targetP>(
                            temp[k][j], acc[k][j], index);
                }
                trgswfftExternalProductbatch<typename bkP::targetP, batch>(temp, temp,
                                                               cs[count]);
                for (int j = 0; j < batch; j++)
                    for (int k = 0; k < bkP::targetP::k + 1; k++)
                        for (int n = 0; n < bkP::targetP::n; n++)
                            acc[k][j][n] += temp[k][j][n];
                count++;
            }
        }
    }
}


}  // namespace TFHEpp