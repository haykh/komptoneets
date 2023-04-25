#ifndef UTILS_PROGRESSBAR_H
#define UTILS_PROGRESSBAR_H

#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include <string_view>

#define IMIN(a, b) ((a) < (b) ? (a) : (b))
#define IMAX(a, b) ((a) > (b) ? (a) : (b))

namespace pbar {
  // averaging window (last 10 % of steps)
  inline constexpr int       average_last_pct { 10 };
  inline constexpr int       width { 37 };
  constexpr std::string_view fill { "■" };
  constexpr std::string_view empty { " " };
  constexpr std::string_view start { "[" };
  constexpr std::string_view end { "]" };
}    // namespace pbar
void ProgressBar(const std::vector<long double>& durations,
                 const double&                   time,
                 const double&                   runtime,
                 std::ostream&                   os = std::cout) {
  // durations are in us (microseconds)
  const auto window
    = IMIN(IMAX(static_cast<int>(durations.size() * pbar::average_last_pct / 100), 10),
           durations.size());
  const auto avg_duration = std::accumulate(durations.end() - window, durations.end(), 0.0)
                            / static_cast<long double>(window);
  auto avg_reduced = avg_duration;
  auto avg_units   = "us";
  if (avg_reduced > 1e6) {
    avg_reduced /= 1e6;
    avg_units = "s";
  } else if (avg_reduced > 1e3) {
    avg_reduced /= 1e3;
    avg_units = "ms";
  } else if (avg_reduced < 1e-2) {
    avg_reduced *= 1e3;
    avg_units = "ns";
  }

  const auto remain_nsteps = static_cast<int>((runtime - time) * durations.size() / time);
  auto       remain_time   = static_cast<long double>(remain_nsteps * avg_duration);
  auto       remain_units  = "us";
  if ((1e3 <= remain_time) && (remain_time < 1e6)) {
    remain_time /= 1e3;
    remain_units = "ms";
  } else if ((1e6 <= remain_time) && (remain_time < 6e7)) {
    remain_time /= 1e6;
    remain_units = "s";
  } else if ((6e7 <= remain_time) && (remain_time < 3.6e9)) {
    remain_time /= 6e7;
    remain_units = "min";
  } else if (3.6e9 <= remain_time) {
    remain_time /= 3.6e9;
    remain_units = "hr";
  }

  const int    nfilled = IMIN(time / runtime * pbar::width, pbar::width);
  const int    nempty  = pbar::width - nfilled;
  const double pct     = time / runtime * static_cast<double>(100.0);
  os << "Average timestep: " << avg_reduced << " " << avg_units << " (last " << window
     << " steps)" << std::endl;
  os << "Remaining time: " << remain_time << " " << remain_units << std::endl;
  os << pbar::start;
  for (std::size_t i { 0 }; i < nfilled; ++i) {
    os << pbar::fill;
  }
  for (std::size_t i { 0 }; i < nempty; ++i) {
    os << pbar::empty;
  }
  os << pbar::end << " " << std::fixed << std::setprecision(2) << pct << "%\n";
  os << "\n\n";
}

#endif    // UTILS_PROGRESSBAR_H