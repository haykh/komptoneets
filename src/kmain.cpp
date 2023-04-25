#include "utils/cargs.h"
#include "utils/progressbar.h"

#include "ksolver.hpp"

#include <Kokkos_Core.hpp>

#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>

using chrono_t = std::chrono::high_resolution_clock;

namespace params {
  const double dt { 0.0002 };
  const int    Nx { 1000 };
  const double T { 0.2 };
  const int    maxsteps { 100000 };
}    // namespace params

auto main(int argc, char* argv[]) -> int {
  Kokkos::initialize(argc, argv);
  {
    CommandLineArguments cl_args;
    cl_args.readCommandLineArguments(argc, argv);
    auto                     output = cl_args.getArgument("-output");

    Simulation               sim(params::dt, params::Nx, params::T);
    std::vector<long double> durations;
    bool                     rewrite { true };

    for (int t { 0 }; t < params::maxsteps; ++t) {
      auto start = chrono_t::now();
      sim.Step();
      if (t % 1000 == 0) {
        sim.Write(output, rewrite);
        rewrite = false;
      }
      auto elapsed = (chrono_t::now() - start).count();
      durations.push_back(elapsed / 1e3);
      sim.set_timestep(t + 1);
      ProgressBar(durations, t * params::dt, params::maxsteps * params::dt);
    }
  }
  Kokkos::finalize();
  return 0;
}