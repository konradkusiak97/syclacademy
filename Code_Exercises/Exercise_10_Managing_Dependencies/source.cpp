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
 * // Do a USM memcpy
 * auto event = q.memcpy(dst_ptr, src_ptr, sizeof(T)*n);
 * // Do a USM memcpy with dependent events
 * auto event = q.memcpy(dst_ptr, src_ptr, sizeof(T)*n, {event1, event2});
 *
 * // Wait on an event
 * event.wait();
 *
 * // Wait on a queue
 * q.wait();
 *
 * // Submit work to the queue
 * auto event = q.submit([&](sycl::handler &cgh) {
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
 * //             i: Without dependent events
 *                    cgh.parallel_for<class mykernel>(sycl::range{n}, 
 *                    [=](sycl::id<1> i) { // Do something });
 * //             ii: With dependent events
 *                    cgh.parallel_for<class mykernel>(sycl::range{n}, 
 *                    {event1, event2}, [=](sycl::id<1> i) { 
 *                        // Do something
 *                      });
 *
*/

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <sycl/sycl.hpp>

class kernelA; 
class kernelB; 
class kernelC; 
class kernelOut; 

TEST_CASE("managing_dependencies", "managing_dependencies_source") {
  constexpr size_t dataSize = 1024;

  int inA[dataSize], inB[dataSize], inC[dataSize], out[dataSize];
  for (int i = 0; i < dataSize; ++i) {
    inA[i] = static_cast<float>(i);
    inB[i] = static_cast<float>(i);
    inC[i] = static_cast<float>(i);
    out[i] = 0.0f;
  }

  try {

    auto queue = sycl::queue{sycl::gpu_selector_v};

    auto buffA = sycl::buffer{inA, sycl::range{dataSize}};
    auto buffB = sycl::buffer{inB, sycl::range{dataSize}};
    auto buffC = sycl::buffer{inC, sycl::range{dataSize}};
    auto buffOut = sycl::buffer{out, sycl::range{dataSize}};

    queue.submit([&](sycl::handler& cgh){
      sycl::accessor accA{buffA, cgh, sycl::read_write};

      cgh.parallel_for<kernelA>(sycl::range{dataSize}, [=](sycl::id<1> idx) {
        accA[idx] = accA[idx] * 2.0f;
      });
    });

    queue.submit([&](sycl::handler& cgh){
      sycl::accessor accA{buffA, cgh, sycl::read_only};
      sycl::accessor accB{buffB, cgh, sycl::write_only};

      cgh.parallel_for<kernelB>(sycl::range{dataSize}, [=](sycl::id<1> idx) {
        accB[idx] += accA[idx];
      });
    });

    queue.submit([&](sycl::handler& cgh){
      sycl::accessor accA{buffA, cgh, sycl::read_only};
      sycl::accessor accC{buffC, cgh, sycl::write_only};

      cgh.parallel_for<kernelC>(sycl::range{dataSize}, [=](sycl::id<1> idx) {
        accC[idx] -= accA[idx];
      });
    });

    queue.submit([&](sycl::handler& cgh){
      sycl::accessor accB{buffB, cgh, sycl::read_only};
      sycl::accessor accC{buffC, cgh, sycl::read_only};
      sycl::accessor accOut{buffOut, cgh, sycl::write_only};

      cgh.parallel_for<kernelOut>(sycl::range{dataSize}, [=](sycl::id<1> idx) {
        accOut[idx] = accB[idx] + accC[idx];
      });
    });

    queue.wait_and_throw();

  } catch (const sycl::exception& e) {
    std::cout << "Exception caught: " << e.what() << std::endl;
  }

  // Task: Run these kernels on the SYCL device, respecting the dependencies
  // as shown in the README

  for (int i = 0; i < dataSize; ++i) {
    REQUIRE(out[i] == i * 2.0f);
  }
}
