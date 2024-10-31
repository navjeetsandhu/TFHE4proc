#pragma once

#include <array>
#include <cstdint>

#include "mulfft.hpp"
#include "params.hpp"
#include "trlwe.hpp"


/*
 * https://www.zama.ai/post/tfhe-deep-dive-part-3
 */

namespace TFHEpp {


// lvl1param  offset = 0x82080000,         2181562368
// lvl2param  offset = 0x8040201000000000, 9241421688455823360
template <class P>
constexpr typename P::T offsetgen()
{
    constexpr uint32_t  max_digits = std::numeric_limits<typename P::T>::digits;
    typename P::T offset = 0;
    // lvl1param   P::l = 3  P::Bgbit 6
    // lvl2param   P::l = 4  P::Bgbit 9
    for (int i = 1; i <= P::l; i++)
        offset += P::Bg / 2 * (1ULL << (max_digits - i * P::Bgbit));
    return offset;
}

template <class P>
inline void Decomposition(DecomposedPolynomial<P> &decpoly,
                          const Polynomial<P> &poly, typename P::T randbits = 0)
{

    constexpr typename P::T offset = offsetgen<P>(); // offset: lvl2 0x8040201000000000 lvl1 0x82080000
                                                     // lvl1   l = 3  Bgbit 6, Bg 64
                                                     // lvl2   l = 4  Bgbit 9, Bg 512

    constexpr uint32_t   roundoffsetBit = std::numeric_limits<typename P::T>::digits - P::l * P::Bgbit - 1;
    std::cout << std::dec << "roundoffsetBit: " << roundoffsetBit << std::endl;
    constexpr typename P::T roundoffset = 1ULL << roundoffsetBit;
    std::cout << std::hex << "roundoffset: " << roundoffset << std::endl;


    constexpr typename P::T mask =
        static_cast<typename P::T>((1ULL << P::Bgbit) - 1);
    std::cout << std::hex << "mask: " << mask << std::endl;


    constexpr typename P::T halfBg = (1ULL << (P::Bgbit - 1));
    std::cout << std::hex << "halfBg: " << halfBg << std::endl;


    for (int i = 0; i < P::n; i++) {
        for (int ii = 0; ii < P::l; ii++)
            decpoly[ii][i] = (((poly[i] + offset + roundoffset) >>
                              (std::numeric_limits<typename P::T>::digits -
                               (ii + 1) * P::Bgbit)) & mask) - halfBg;
    }
}

template <class P>
void DecompositionNTT(DecomposedPolynomialNTT<P> &decpolyntt,
                      const Polynomial<P> &poly, typename P::T randbits = 0)
{
    DecomposedPolynomial<P> decpoly;
    Decomposition<P>(decpoly, poly);
    for (int i = 0; i < P::l; i++) TwistINTT<P>(decpolyntt[i], decpoly[i]);
}

template <class P>
void DecompositionRAINTT(DecomposedPolynomialRAINTT<P> &decpolyntt,
                         const Polynomial<P> &poly, typename P::T randbits = 0)
{
    DecomposedPolynomial<P> decpoly;
    Decomposition<P>(decpoly, poly);
    for (int i = 0; i < P::l; i++)
        raintt::TwistINTT<typename P::T, P::nbit, false>(
            decpolyntt[i], decpoly[i], (*raintttable)[1], (*raintttwist)[1]);
}

}  // namespace TFHEpp