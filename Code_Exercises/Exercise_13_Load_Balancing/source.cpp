/*
 SYCL Academy (c)

 SYCL Academy is licensed under a Creative Commons
 Attribution-ShareAlike 4.0 International License.

 You should have received a copy of the license along with this
 work.  If not, see <http://creativecommons.org/licenses/by-sa/4.0/>.

 * SYCL Quick Reference
 * ~~~~~~~~~~~~~~~~~~~~
 *
 * // Get all available devices
 * auto devs = sycl::device::get_devices();
 *
 * // Construct a queue with a device
 * auto q = sycl::queue{my_device};
 *
 * // Declare a buffer pointing to ptr
 * auto buf = sycl::buffer{ptr, sycl::range{n}};
 *
 * // Submit work to the queue
 * q.submit([&](sycl::handler &cgh) {
 *   // COMMAND GROUP
 * });
 *
 * // Within the command group you can
 * //    1. Declare an accessor to a buffer
 *          auto read_write_acc = sycl::accessor{buf, cgh};
 *          auto read_acc = sycl::accessor{buf, cgh, sycl::read_only};
 *          auto write_acc = sycl::accessor{buf, cgh, sycl::write_only};
 *          auto no_init_acc = sycl::accessor{buf, cgh, sycl::no_init};
 * //    2. Enqueue a single task:
 *          cgh.single_task<class mykernel>([=]() {
 *              // Do something
 *          });
 * //    3. Enqueue a parallel for:
 *          cgh.parallel_for<class mykernel>(sycl::range{n}, [=](sycl::id<1> i) {
 *              // Do something
 *          });
 *

*/

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <sycl/sycl.hpp>

class vector_add_first;
class vector_add_second;

std::vector<sycl::device> get_two_devices() {
  auto devs = sycl::device::get_devices();

  if (devs.size() == 1) { return {devs[0], devs[0]}; }
  else { return {devs[1], devs[2]}; }; 
}

TEST_CASE("load_balancing", "load_balancing_source") {
  constexpr size_t dataSize = 1024;
  constexpr float ratio = 0.5f;
  constexpr size_t dataSizeFirst = ratio * dataSize;
  constexpr size_t dataSizeSecond = dataSize - dataSizeFirst;

  float a[dataSize], b[dataSize], r[dataSize];
  for (int i = 0; i < dataSize; ++i) {
    a[i] = static_cast<float>(i);
    b[i] = static_cast<float>(i);
    r[i] = 0.0f;
  }

  try {
    std::vector<sycl::device> devices = get_two_devices();  

    auto queueDev1 = sycl::queue{devices[0]};
    auto queueDev2 = sycl::queue{devices[1]};

    std::cout << "Running on devices:" << std::endl;
    std::cout << "1: " << queueDev1.get_device().get_info<sycl::info::device::name>() << std::endl;
    std::cout << "2: " << queueDev2.get_device().get_info<sycl::info::device::name>() << std::endl;

    auto bufFirstA = sycl::buffer{a, sycl::range{dataSizeFirst}};
    auto bufFirstB = sycl::buffer{b, sycl::range{dataSizeFirst}};
    auto bufFirstR = sycl::buffer{r, sycl::range{dataSizeFirst}};

    auto bufSecondA =
        sycl::buffer{a + dataSizeFirst, sycl::range{dataSizeSecond}};
    auto bufSecondB =
        sycl::buffer{b + dataSizeFirst, sycl::range{dataSizeSecond}};
    auto bufSecondR =
        sycl::buffer{r + dataSizeFirst, sycl::range{dataSizeSecond}};

    queueDev1.submit([&](sycl::handler &cgh) {
      sycl::accessor accA{bufFirstA, cgh, sycl::read_only};
      sycl::accessor accB{bufFirstB, cgh, sycl::read_only};
      sycl::accessor accR{bufFirstR, cgh, sycl::write_only};

      cgh.parallel_for<vector_add_first>(
          sycl::range{dataSizeFirst},
          [=](sycl::id<1> idx) { accR[idx] = accA[idx] + accB[idx]; });
    });

    queueDev2.submit([&](sycl::handler &cgh) {
      sycl::accessor accA{bufSecondA, cgh, sycl::read_only};
      sycl::accessor accB{bufSecondB, cgh, sycl::read_only};
      sycl::accessor accR{bufSecondR, cgh, sycl::write_only};

      cgh.parallel_for<vector_add_second>(
          sycl::range{dataSizeSecond},
          [=](sycl::id<1> idx) { accR[idx] = accA[idx] + accB[idx]; });
    });

    queueDev1.wait_and_throw();
    queueDev2.wait_and_throw();

  } catch (const sycl::exception& e) {
    std::cout << "Exception caught: " << e.what() << std::endl;
  }
  // Task: split the total work across two distinct SYCL devices
  // You might split the work as in the two loops below.

  for (int i = 0; i < dataSize; ++i) {
    REQUIRE(r[i] == static_cast<float>(i) * 2.0f);
  }
}
