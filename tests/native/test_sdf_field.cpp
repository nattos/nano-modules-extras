// The support matrix for the `sdf_field` provider contract, executably.
// fx::sdf_field::validate() is the single source of truth for which
// leaf combinations a consumer will render; every branch is pinned here
// so growing support (GridOnly, Volumetric, variable grid_ext, …) means
// consciously flipping a case from rejected to accepted.

#include <catch2/catch_test_macros.hpp>

#include <cstring>

#include "effect_sdf_field.h"

using fx::sdf_field::Class;
using fx::sdf_field::Desc;
using fx::sdf_field::kGridExt;
using fx::sdf_field::validate;

namespace {

// The one combination supported today: spherical heightmap with grid +
// shell textures and every scalar explicitly declared.
Desc valid_spherical_heightmap() {
  Desc d;
  d.field_class = Class::SphericalHeightmap;
  d.radius = 0.5f;
  d.lip = 0.4f;
  d.lip_true = 0.25f;
  d.crest_amp = 0.2f;
  d.crest_gain = 1.0f;
  d.grid_ext = kGridExt;
  d.shell_res = 1024.f;
  d.has_grid = true;
  d.has_shell = true;
  return d;
}

bool rejected_mentioning(const Desc& d, const char* needle) {
  Class c = Class::SphericalHeightmap;
  const char* reason = validate(d, &c);
  return reason != nullptr && c == Class::None &&
         std::strstr(reason, needle) != nullptr;
}

}  // namespace

TEST_CASE("spherical heightmap with all fields is the supported combo",
          "[sdf_field]") {
  Desc d = valid_spherical_heightmap();
  Class c = Class::None;
  REQUIRE(validate(d, &c) == nullptr);
  REQUIRE(c == Class::SphericalHeightmap);
}

TEST_CASE("lip floored above lip_true is valid (band-widen contract)",
          "[sdf_field]") {
  // The provider may floor lip for march speed; lip_true == lip is the
  // unfloored case and also valid.
  Desc d = valid_spherical_heightmap();
  d.lip_true = d.lip;
  Class c = Class::None;
  REQUIRE(validate(d, &c) == nullptr);
  REQUIRE(c == Class::SphericalHeightmap);
}

TEST_CASE("crest channel may be declared inert", "[sdf_field]") {
  Desc d = valid_spherical_heightmap();
  d.crest_gain = 0.f;   // .b inert — consumer must not shade it
  d.crest_amp = 0.f;    // smooth sphere
  Class c = Class::None;
  REQUIRE(validate(d, &c) == nullptr);
}

TEST_CASE("declared-but-unsupported classes are rejected by name",
          "[sdf_field]") {
  Desc d = valid_spherical_heightmap();
  d.field_class = Class::GridOnly;
  REQUIRE(rejected_mentioning(d, "GridOnly"));
  d.field_class = Class::Volumetric;
  REQUIRE(rejected_mentioning(d, "Volumetric"));
  d.field_class = Class::None;
  REQUIRE(rejected_mentioning(d, "no field class"));
  d.field_class = 42;
  REQUIRE(rejected_mentioning(d, "unknown"));
}

TEST_CASE("spherical heightmap requires both textures", "[sdf_field]") {
  Desc d = valid_spherical_heightmap();
  d.has_grid = false;
  REQUIRE(rejected_mentioning(d, "grid texture"));
  d = valid_spherical_heightmap();
  d.has_shell = false;
  REQUIRE(rejected_mentioning(d, "shell texture"));
}

TEST_CASE("scalar declarations are range-checked", "[sdf_field]") {
  Desc d = valid_spherical_heightmap();
  d.radius = 0.f;
  REQUIRE(rejected_mentioning(d, "radius"));

  d = valid_spherical_heightmap();
  d.lip = 0.f;
  REQUIRE(rejected_mentioning(d, "lip must be in"));
  d.lip = 1.5f;
  REQUIRE(rejected_mentioning(d, "lip must be in"));

  d = valid_spherical_heightmap();
  d.lip_true = 0.f;
  REQUIRE(rejected_mentioning(d, "lip_true"));
  d.lip_true = d.lip + 0.1f;   // grid would UNDERSTATE distances — unsafe
  REQUIRE(rejected_mentioning(d, "lip_true"));

  d = valid_spherical_heightmap();
  d.crest_amp = -0.1f;
  REQUIRE(rejected_mentioning(d, "crest_amp"));

  d = valid_spherical_heightmap();
  d.shell_res = 32.f;
  REQUIRE(rejected_mentioning(d, "shell_res"));
}

TEST_CASE("dust is optional and gated on its buffer", "[sdf_field]") {
  // No dust at all — the pre-dust rail, still the baseline combo.
  Desc d = valid_spherical_heightmap();
  Class c = Class::None;
  REQUIRE(validate(d, &c) == nullptr);

  // A parked pool (buffer present, count 0) is valid and means "no dust".
  d = valid_spherical_heightmap();
  d.has_dust = true;
  REQUIRE(validate(d, &c) == nullptr);
  REQUIRE(c == Class::SphericalHeightmap);

  // Live dust requires the buffer.
  d = valid_spherical_heightmap();
  d.dust_count = 1000;
  REQUIRE(rejected_mentioning(d, "dust buffer"));
  d.has_dust = true;
  REQUIRE(validate(d, &c) == nullptr);

  // Counts are range-checked.
  d = valid_spherical_heightmap();
  d.has_dust = true;
  d.dust_count = -1;
  REQUIRE(rejected_mentioning(d, "dust_count"));
  d.dust_count = fx::sdf_field::kDustMax + 1;
  REQUIRE(rejected_mentioning(d, "kDustMax"));
  d.dust_count = fx::sdf_field::kDustMax;
  REQUIRE(validate(d, &c) == nullptr);
}

TEST_CASE("v1 grid conventions are enforced", "[sdf_field]") {
  // Consumer shaders bake extent/res as compile-time constants, so a
  // mismatched provider must be refused, not mis-sampled.
  Desc d = valid_spherical_heightmap();
  d.grid_ext = 1.0f;
  REQUIRE(rejected_mentioning(d, "grid_ext"));

  // The surface (crest sphere) must fit inside the declared grid.
  d = valid_spherical_heightmap();
  d.radius = 0.7f;
  d.crest_amp = 0.3f;
  REQUIRE(rejected_mentioning(d, "crest sphere"));
}
