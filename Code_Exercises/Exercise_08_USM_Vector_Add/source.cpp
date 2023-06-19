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

class usm_add;

// USM selector
int usm_selector(const sycl::device& dev) {
  return dev.has(sycl::aspect::usm_device_allocations);
}

TEST_CASE("usm_vector_add", "usm_vector_add_source") {
  constexpr size_t dataSize = 1024;

  float a[dataSize], b[dataSize], r[dataSize];
  for (int i = 0; i < dataSize; ++i) {
    a[i] = static_cast<float>(i);
    b[i] = static_cast<float>(i);
    r[i] = 0.0f;
  }

  try {

    // Task: Allocate the arrays in USM, and compute r[i] = a[i] + b[i] on the SYCL device
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

    queue.memcpy(a, device_ptrA, dataSize * sizeof(float)).wait();
    queue.memcpy(b, device_ptrB, dataSize * sizeof(float)).wait();
    queue.memcpy(r, device_ptrR, dataSize * sizeof(float)).wait();

    sycl::free(device_ptrA, queue);
    sycl::free(device_ptrB, queue);
    sycl::free(device_ptrR, queue);

    queue.throw_asynchronous();
  } catch(const sycl::exception &e) {
    std::cout << "Exception caught: " << e.what() << std::endl;
  }

  for (int i = 0; i < dataSize; ++i) {
    REQUIRE(r[i] == i * 2);
  }
}
