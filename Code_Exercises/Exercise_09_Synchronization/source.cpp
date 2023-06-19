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
 * // Do a USM malloc_device
 * auto ptr = sycl::malloc_device<T>(n, q);
 *
 * // Do a USM memcpy
 * q.memcpy(dst_ptr, src_ptr, sizeof(T)*n);
 *
 * // Wait on a queue
 * q.wait();
 *
 * // Submit work to the queue
 * q.submit([&](sycl::handler &cgh) {
 *   // COMMAND GROUP
 * });
 *
 *
 * // Within the command group you can
 * //    1. Declare an accessor to a buffer
 *          auto read_write_acc = sycl::accessor{buf, cgh};
 *          auto read_acc = sycl::accessor{buf, cgh, sycl::read_only};
 *          auto write_acc = sycl::accessor{buf, cgh, sycl::write_only};
 *          auto no_init_acc = sycl::accessor{buf, cgh, sycl::no_init};
 * //    2. Enqueue a parallel for:
 *              cgh.parallel_for<class mykernel>(sycl::range{n}, 
 *                    [=](sycl::id<1> i) { // Do something });
 *
 *
*/

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include <sycl/sycl.hpp>

class usm_add;
class vector_add;

// USM selector
int usm_selector(const sycl::device& dev) {
  return dev.has(sycl::aspect::usm_device_allocations);
}

TEST_CASE("synchronization_usm", "synchronization_source") {
  // Use your code from Exercise 3 to start
  constexpr size_t dataSize = 1024;

  float a[dataSize], b[dataSize], r[dataSize];
  for (int i = 0; i < dataSize; ++i) {
    a[i] = static_cast<float>(i);
    b[i] = static_cast<float>(i);
    r[i] = 0.0f;
  }

  try {
    auto queue = sycl::queue{usm_selector}; 

    auto* device_ptrA = sycl::malloc_device<float>(dataSize, queue); 
    auto* device_ptrB = sycl::malloc_device<float>(dataSize, queue); 
    auto* device_ptrR = sycl::malloc_device<float>(dataSize, queue); 

    queue.memcpy(device_ptrA, a, dataSize * sizeof(float)).wait();
    queue.memcpy(device_ptrB, b, dataSize * sizeof(float)).wait();
    queue.memcpy(device_ptrR, r, dataSize * sizeof(float)).wait();

    queue.parallel_for<usm_add>(sycl::range(dataSize), [=](sycl::id<1> idx) {
      device_ptrR[idx] = device_ptrA[idx] + device_ptrB[idx];

    }).wait();

    queue.memcpy(a, device_ptrA, dataSize * sizeof(float));
    queue.memcpy(b, device_ptrB, dataSize * sizeof(float));
    queue.memcpy(r, device_ptrR, dataSize * sizeof(float));

    queue.wait_and_throw();

    sycl::free(device_ptrA, queue);
    sycl::free(device_ptrB, queue);
    sycl::free(device_ptrR, queue);

  } catch(const sycl::exception &e) {
    std::cout << "Exception caught: " << e.what() << std::endl;
  }

  for (int i = 0; i < dataSize; ++i) {
    REQUIRE(r[i] == i * 2);
  }
  REQUIRE(true);
}

TEST_CASE("synchronization_buffer_acc", "synchronization_source") {
  // Use your code from Exercise 3 to start
  constexpr size_t dataSize = 1024;

  float a[dataSize], b[dataSize], r[dataSize];
  for (int i = 0; i < dataSize; ++i) {
    a[i] = static_cast<float>(i);
    b[i] = static_cast<float>(i);
    r[i] = 0.0f;
  }

  try {

    auto queue = sycl::queue{sycl::gpu_selector_v};
    std::cout << "Chosen device: "
              << queue.get_device().get_info<sycl::info::device::name>()
              << std::endl;

    {
      auto bufA = sycl::buffer{a, sycl::range{dataSize}};
      auto bufB = sycl::buffer{b, sycl::range{dataSize}};
      auto bufR = sycl::buffer{r, sycl::range{dataSize}};

      queue.submit([&](sycl::handler &cgh) {
        auto accA = sycl::accessor{bufA, cgh, sycl::read_only};
        auto accB = sycl::accessor{bufB, cgh, sycl::read_only};
        auto accR = sycl::accessor{bufR, cgh, sycl::write_only};

        cgh.parallel_for<vector_add>(sycl::range<1>(dataSize), [=] (sycl::id<1> idx) {

            accR[idx] = accA[idx] + accB[idx];
        });

      }).wait();

      {
        auto hostAccR = bufR.get_host_access(sycl::read_only);
        for (int i = 0; i < dataSize; ++i) {
          REQUIRE(hostAccR[i] == i * 2);
        }
      }

    } // Copy data back
    
    queue.throw_asynchronous();
  } catch (const sycl::exception &e) {
    std::cout << "Exception caught: " << e.what() << std::endl;
  }

  for (int i = 0; i < dataSize; ++i) {
    REQUIRE(r[i] == static_cast<float>(i) * 2.0f);
  }
  REQUIRE(true);
}
