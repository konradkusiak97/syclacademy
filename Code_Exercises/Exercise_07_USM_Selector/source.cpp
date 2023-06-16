/*
 SYCL Academy (c)

 SYCL Academy is licensed under a Creative Commons
 Attribution-ShareAlike 4.0 International License.

 You should have received a copy of the license along with this
 work.  If not, see <http://creativecommons.org/licenses/by-sa/4.0/>.
*/

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <sycl/sycl.hpp>

int usm_selector(const sycl::device& dev) {
  return dev.has(sycl::aspect::usm_device_allocations);
}

TEST_CASE("usm_selector", "usm_selector_source") {

  // Task: create a queue to a device which supports USM allocations
  // Remember to check for exceptions

  try {
    auto usmQueue  = sycl::queue{usm_selector};

    std::cout << "Chosen device: " << usmQueue.get_device().get_info<sycl::info::device::name>() << std::endl;

    usmQueue.throw_asynchronous();
  } catch (const sycl::exception& e) {
    std::cout << "Exception caught: " << e.what() << std::endl;
  }

  REQUIRE(true);
}
