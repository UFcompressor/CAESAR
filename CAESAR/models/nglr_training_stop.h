#pragma once

namespace nglr::detail {

class ZeroLossStop {
public:
  bool observe(double epoch_loss) {
    consecutive_ = epoch_loss == 0.0 ? consecutive_ + 1 : 0;
    return consecutive_ >= 3;
  }

private:
  int consecutive_ = 0;
};

} // namespace nglr::detail
