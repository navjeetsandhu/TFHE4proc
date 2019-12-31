#pragma once

#include <array>
#include <params.hpp>

namespace TFHEpp {
using namespace std;

void IdentityKeySwitchlvl10(TLWElvl0 &res, TLWElvl1 &tlwe,
                            const KeySwitchingKey &ksk);
}  // namespace TFHEpp