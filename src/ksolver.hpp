#ifndef KSOLVER_HPP
#define KSOLVER_HPP

#include <Kokkos_Core.hpp>

#include <filesystem>
#include <fstream>

#define SQR(x) ((x) * (x))

namespace math = Kokkos;
using array_t  = Kokkos::View<double*>;

struct First {};
struct Second {};

/**
 * @brief      Functor to compute the first and second derivative of a function
 *             using a 2nd order finite difference scheme.
 * @details    Assumes a uniform grid and n''(x) = 0 at the boundaries.
 * @note       Usage:
 * ... Kokkos::parallel_for("name", Kokkos::RangePolicy<First>(0, Nx), Derivative(n, Dn, dx));
 */
class Derivative {
  array_t           n;
  array_t           Dn;
  const double      dx;
  const std::size_t imax;

public:
  Derivative(array_t& n_, array_t& Dn_, const double& dx_)
    : n { n_ }, Dn { Dn_ }, dx { dx_ }, imax { n.extent(0) - 1 } {}

  KOKKOS_FUNCTION
  void operator()(First, const std::size_t i) const {
    if (i == 0) {
      Dn(i) = ((-7.0 / 2.0) * n(1) + 5.0 * n(2) - 1.5 * n(3)) / (2.0 * dx);
    } else if (i == imax) {
      Dn(i) = ((7.0 / 2.0) * n(imax - 1) - 5.0 * n(imax - 2) + 1.5 * n(imax - 3)) / (2.0 * dx);
    } else {
      Dn(i) = (n(i + 1) - n(i - 1)) / (2.0 * dx);
    }
  }

  KOKKOS_FUNCTION
  void operator()(Second, const std::size_t i) const {
    if (i == 0 || i == imax) {
      Dn(i) = 0.0;
    } else {
      Dn(i) = (n(i + 1) - 2.0 * n(i) + n(i - 1)) / SQR(dx);
    }
  }
};

class Simulation {
  array_t           x;
  array_t           n;
  array_t           n1;
  array_t           dn1_dx;
  array_t           d2n1_dx2;
  double            dx;
  const double      dt;
  const double      T;
  const std::size_t Nx;
  int               tstep { 0 };
  const double      T0 { 1e-2 };
  const double      e_min { 1e-4 }, e_max { 10.0 };

public:
  Simulation(const double& dt_, const std::size_t& Nx_, const double& T_)
    : dt { dt_ }, Nx { Nx_ }, T { T_ } {
    x                  = array_t("x", Nx);
    n                  = array_t("n", Nx);
    n1                 = array_t("n1", Nx);
    dn1_dx             = array_t("dn1/dx", Nx);
    d2n1_dx2           = array_t("d2n1/dx2", Nx);
    const double x_min = math::log(e_min), x_max = math::log(e_max);
    Kokkos::parallel_for(
      "init", Nx, KOKKOS_LAMBDA(const std::size_t i) {
        // grid
        x(i) = x_min + (x_max - x_min) * static_cast<double>(i) / static_cast<double>(Nx - 1);
        const double e = math::exp(x(i));
        // initial condition
        if (math::fabs(e - T0) < 1e-3) {
          n(i) = 1.0;
        } else {
          n(i) = 0.0;
        }
      });
    dx = (x_max - x_min) / static_cast<double>(Nx);
  }

  ~Simulation() = default;

  [[nodiscard]] auto timestep() const -> int {
    return tstep;
  }

  [[nodiscard]] auto time() const -> double {
    return tstep * dt;
  }

  void set_timestep(const int& t) {
    tstep = t;
  }

  void Step() {
    Kokkos::deep_copy(n1, n);
    {
      ComputeDerivatives();
      Iterate();
      ComputeDerivatives();
      Iterate();
      ComputeDerivatives();
      Iterate();
    }
    ComputeDerivatives();
    Final();
  }

  void ComputeDerivatives() {
    Kokkos::parallel_for(
      "first", Kokkos::RangePolicy<First>(0, Nx), Derivative(n1, dn1_dx, dx));
    Kokkos::parallel_for(
      "second", Kokkos::RangePolicy<Second>(0, Nx), Derivative(n1, d2n1_dx2, dx));
  }

  KOKKOS_INLINE_FUNCTION
  double RHS(const std::size_t i) const {
    const double e = math::exp(x(i));
    return T * d2n1_dx2(i) + (e * (1.0 + 2.0 * n1(i)) + 3.0 * T) * dn1_dx(i)
           + 4.0 * e * n1(i) * (1.0 + n1(i));
  }

  void Iterate() {
    // n1 = n + dt * RHS(n1)
    // n1 = 0.5 * (n + n1)
    Kokkos::parallel_for(
      "update", Nx, KOKKOS_LAMBDA(const std::size_t i) {
        // predictor
        n1(i) = n(i) + dt * RHS(i);
        // corrector
        n1(i) = 0.5 * (n(i) + n1(i));
        n1(i) = math::fabs(n1(i));
      });
  }

  void Final() {
    // n = n + dt * RHS(n1)
    Kokkos::parallel_for(
      "final", Nx, KOKKOS_LAMBDA(const std::size_t i) {
        n(i) += dt * RHS(i);
        n(i) = math::fabs(n(i));
      });
  }

  void Write(const std::string_view& output, const bool& rewrite = false) {
    if (rewrite) {
      if (std::filesystem::exists(output)) {
        std::filesystem::remove_all(output);
      }
      std::filesystem::create_directory(output);
    }

    std::ofstream file;
    file.open(std::string(output) + "/time_" + std::to_string(tstep) + ".csv");
    file << "## time:" << tstep * dt << "\n";
    file << "## temperature: " << T << "\n";
    file << "# e,n\n";
    auto n_h = Kokkos::create_mirror_view(n);
    Kokkos::deep_copy(n_h, n);
    auto x_h = Kokkos::create_mirror_view(x);
    Kokkos::deep_copy(x_h, x);
    for (std::size_t i { 0 }; i < Nx; ++i) {
      file << math::exp(x_h(i)) << "," << n_h(i) << "\n";
    }
    file.close();
  }
};

#endif    // KSOLVER_HPP