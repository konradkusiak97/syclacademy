/*
 SYCL Academy (c)

 SYCL Academy is licensed under a Creative Commons
 Attribution-ShareAlike 4.0 International License.

 You should have received a copy of the license along with this
 work.  If not, see <http://creativecommons.org/licenses/by-sa/4.0/>.

 * SYCL Quick Reference
 * ~~~~~~~~~~~~~~~~~~~~
 *
 * // Default construct a queue
 * auto q = sycl::queue{};
 *
 * // Declare a buffer pointing to ptr
 * auto buf = sycl::buffer{ptr, sycl::range{n}};
 *
 * // Declare a buffer using host ptr
 * auto buf = sycl::buffer{ptr, sycl::range{n}, 
 *                 {sycl::property::buffer::use_host_ptr{}}};
 *
 * // Declare a buffer relating to no host memory
 * auto buf = sycl::buffer{sycl::range{n}};
 *
 * // Set final data of a buffer
 * buf.set_final_data(host_ptr);
 *
 * // Set final data of a buffer to nullptr
 * buf.set_final_data(nullptr);
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
 * //    2. Enqueue a parallel for:
 *          cgh.parallel_for<class mykernel>(sycl::range{n}, [=](sycl::id<1> i) {
 *              // Do something
 *          });
 *
*/

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <sycl/sycl.hpp>

class kernel1;
class kernel2;

TEST_CASE("temporary_data", "temporary_data_source") {
  constexpr size_t dataSize = 1024;

  float in[dataSize], out[dataSize], tmp[dataSize];
  for (int i = 0; i < dataSize; ++i) {
    in[i] = static_cast<float>(i);
    tmp[i] = 0.0f;
    out[i] = 0.0f;
  }

  try {
    auto queue = sycl::queue{sycl::gpu_selector_v};

    auto buffIn = sycl::buffer{in, sycl::range{dataSize}};
    auto buffTmp = sycl::buffer<float>{sycl::range{dataSize}};

    buffIn.set_final_data(nullptr);
    buffTmp.set_final_data(out);

    queue.submit([&](sycl::handler& cgh) {
      auto accIn = buffIn.get_access(cgh);
      auto accTmp = buffTmp.get_access(cgh);

      cgh.parallel_for<kernel1>(sycl::range{dataSize}, [=](sycl::id<1> idx) {
        accTmp[idx] = accIn[idx] * 8.0f;
      });
    });

    queue.submit([&](sycl::handler& cgh) {
      auto accTmp = buffTmp.get_access(cgh);

      cgh.parallel_for<kernel2>(sycl::range{dataSize}, [=](sycl::id<1> idx) {
        accTmp[idx] /= 2.0f;
      });
    });

    queue.wait_and_throw();
  } catch (const sycl::exception &e) {
    std::cout << "Exception caught: " << e.what() << std::endl;
  }

  // Task: run these kernels on a SYCL device, minimising the memory transfers between the host and device

  for (int i = 0; i < dataSize; ++i) {
    REQUIRE(out[i] == i * 4.0f);
  }
}
