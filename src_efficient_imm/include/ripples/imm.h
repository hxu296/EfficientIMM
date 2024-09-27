//===------------------------------------------------------------*- C++ -*-===//
//
//             Ripples: A C++ Library for Influence Maximization
//                  Marco Minutoli <marco.minutoli@pnnl.gov>
//                   Pacific Northwest National Laboratory
//
//===----------------------------------------------------------------------===//
//
// Copyright (c) 2019, Battelle Memorial Institute
//
// Battelle Memorial Institute (hereinafter Battelle) hereby grants permission
// to any person or entity lawfully obtaining a copy of this software and
// associated documentation files (hereinafter “the Software”) to redistribute
// and use the Software in source and binary forms, with or without
// modification.  Such person or entity may use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and may permit
// others to do so, subject to the following conditions:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimers.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Other than as used herein, neither the name Battelle Memorial Institute or
//    Battelle may be used in any form whatsoever without the express written
//    consent of Battelle.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL BATTELLE OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
//===----------------------------------------------------------------------===//

#ifndef RIPPLES_IMM_H
#define RIPPLES_IMM_H

#include <cmath>
#include <cstddef>
#include <limits>
#include <unordered_map>
#include <vector>

#include <numa.h>
#include <numaif.h>
#include <unistd.h>
#include <sys/mman.h>

#include "nlohmann/json.hpp"
#include "trng/lcg64.hpp"
#include "trng/uniform01_dist.hpp"
#include "trng/uniform_int_dist.hpp"

#include "ripples/configuration.h"
//#include "ripples/find_most_influential.h"
#include "ripples/generate_rrr_sets.h"
#include "ripples/imm_execution_record.h"
#include "ripples/tim.h"
#include "ripples/utility.h"
#include "ripples/find_most_influential_atomics.h"

// Comment out papi
// #include <papi.h>
#include "ripples/streaming_rrr_generator.h"

#define CUDA_PROFILE 0

// For Bitmap
#define BYTE_TO_BIT 8

namespace ripples {

//! The IMM algorithm configuration descriptor.
struct IMMConfiguration : public TIMConfiguration {
  size_t streaming_workers{0};
  size_t streaming_gpu_workers{0};
  size_t seed_select_max_workers{std::numeric_limits<size_t>::max()};
  size_t seed_select_max_gpu_workers{0};
  std::string gpu_mapping_string{""};
  std::unordered_map<size_t, size_t> worker_to_gpu;

  //! \brief Add command line options to configure IMM.
  //!
  //! \param app The command-line parser object.
  void addCmdOptions(CLI::App &app) {
    TIMConfiguration::addCmdOptions(app);
    app.add_option(
           "--streaming-gpu-workers", streaming_gpu_workers,
           "The number of GPU workers for the CPU+GPU streaming engine.")
        ->group("Streaming-Engine Options");
    app.add_option("--streaming-gpu-mapping", gpu_mapping_string,
                   "A comma-separated set of OpenMP numbers for GPU workers.")
        ->group("Streaming-Engine Options");
    app.add_option("--seed-select-max-workers", seed_select_max_workers,
                   "The max number of workers for seed selection.")
        ->group("Streaming-Engine Options");
    app.add_option("--seed-select-max-gpu-workers", seed_select_max_gpu_workers,
                   "The max number of GPU workers for seed selection.")
        ->group("Streaming-Engine Options");
  }
};

//! Retrieve the configuration parsed from command line.
//! \return the configuration parsed from command line.
ToolConfiguration<ripples::IMMConfiguration> configuration();

//! Approximate logarithm of n chose k.
//! \param n
//! \param k
//! \return an approximation of log(n choose k).
inline double logBinomial(size_t n, size_t k) {
  return n * log(n) - k * log(k) - (n - k) * log(n - k);
}

//! Compute ThetaPrime.
//!
//! \tparam execution_tag The execution policy
//!
//! \param x The index of the current iteration.
//! \param epsilonPrime Parameter controlling the approximation factor.
//! \param l Parameter usually set to 1.
//! \param k The size of the seed set.
//! \param num_nodes The number of nodes in the input graph.
template <typename execution_tag>
ssize_t ThetaPrime(ssize_t x, double epsilonPrime, double l, size_t k,
                   size_t num_nodes, execution_tag &&) {
  k = std::min(k, num_nodes/2);
  return (2 + 2. / 3. * epsilonPrime) *
         (l * std::log(num_nodes) + logBinomial(num_nodes, k) +
          std::log(std::log2(num_nodes))) *
         std::pow(2.0, x) / (epsilonPrime * epsilonPrime);
}

//! Compute Theta.
//!
//! \param epsilon Parameter controlling the approximation factor.
//! \param l Parameter usually set to 1.
//! \param k The size of the seed set.
//! \param LB The estimate of the lower bound.
//! \param num_nodes The number of nodes in the input graph.
inline size_t Theta(double epsilon, double l, size_t k, double LB,
                    size_t num_nodes) {
  if (LB == 0) return 0;

  k = std::min(k, num_nodes/2);
  double term1 = 0.6321205588285577;  // 1 - 1/e
  double alpha = sqrt(l * std::log(num_nodes) + std::log(2));
  double beta = sqrt(term1 * (logBinomial(num_nodes, k) +
                              l * std::log(num_nodes) + std::log(2)));
  double lamdaStar = 2 * num_nodes * (term1 * alpha + beta) *
                     (term1 * alpha + beta) * pow(epsilon, -2);

  // std::cout << "#### " << lamdaStar << " / " << LB << " = " << lamdaStar / LB << std::endl;
  return lamdaStar / LB;
}


unsigned long omp_get_thread_num_wrapper(void){
    return (unsigned long)omp_get_thread_num();
}
//! Collect a set of Random Reverse Reachable set.
//!
//! \tparam GraphTy The type of the input graph.
//! \tparam RRRGeneratorTy The type of the RRR generator.
//! \tparam diff_model_tag Type-Tag to selecte the diffusion model.
//! \tparam execution_tag Type-Tag to select the execution policy.
//!
//! \param G The input graph.  The graph is transoposed.
//! \param k The size of the seed set.
//! \param epsilon The parameter controlling the approximation guarantee.
//! \param l Parameter usually set to 1.
//! \param generator The rrr sets generator.
//! \param record Data structure storing timing and event counts.
//! \param model_tag The diffusion model tag.
//! \param ex_tag The execution policy tag.
template <typename GraphTy, typename ConfTy, typename RRRGeneratorTy,
          typename diff_model_tag, typename execution_tag>
auto Sampling(const GraphTy &G, long* counter, const ConfTy &CFG, double l,
              RRRGeneratorTy &generator, IMMExecutionRecord &record,
              diff_model_tag &&model_tag, execution_tag &&ex_tag) {
  using vertex_type = typename GraphTy::vertex_type;
  size_t k = CFG.k;
  double epsilon = CFG.epsilon;

  // sqrt(2) * epsilon
  double epsilonPrime = 1.4142135623730951 * epsilon;

  double LB = 0;
  // std::vector<std::vector<bool>> RR;
  using AdaptiveRRRType = std::variant<std::unique_ptr<std::vector<uint64_t>>, std::unique_ptr<std::vector<bool>>>;
  std::vector<AdaptiveRRRType> RR;

  auto start = std::chrono::high_resolution_clock::now();
  size_t thetaPrime = 0;

  // Set the bitmap 
  
  
  long pg_size = sysconf(_SC_PAGESIZE);
  int num_procs = omp_get_num_procs();

  // TODO: When the number of interleaved numa nodes is less than total, this will cause issue
  // TODO: For hyperthreading, this will cause issue (need to do something like:omp_get_max_threads() % num_procs)
  size_t num_numas =  numa_num_configured_nodes();
  size_t threads_per_numa = (omp_get_max_threads() + num_numas - 1) / num_numas;
  // The bitmap size will be num_vert / 8. (When we use unsigned long as the representation for bits)
  size_t num_vert = G.num_nodes();
  size_t bits_per_entry = sizeof(size_t) * BYTE_TO_BIT;
  size_t bitmap_size = (num_vert + bits_per_entry - 1) / bits_per_entry;
  // Readjust the bitmap_size to byte
  bitmap_size *= sizeof(size_t);
  size_t region_size = threads_per_numa * bitmap_size;
  size_t page_multiple = (region_size + pg_size - 1) / pg_size;
  // Readjust the region_size
  region_size = page_multiple * pg_size;
  size_t map_size = region_size * num_numas;
  void* raw_bitmap = mmap(NULL, map_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (raw_bitmap == MAP_FAILED) {
      perror("mmap failed");
      exit(1);
  }
  // Cast the void to unsigned long which is size_t to do the 
  size_t* mapped_bitmap = static_cast<size_t*>(raw_bitmap);

  // Bind to NUMA nodes
  for(size_t i = 0; i < num_numas; ++i) {
    size_t strt_numa_addr = region_size * i;
    // Shift to get the node_mask
    size_t node_mask = 1UL << i;  
    if (mbind((char*)mapped_bitmap + strt_numa_addr, region_size, MPOL_BIND, &node_mask, 64, MPOL_MF_STRICT)) {
        perror("mbind failed");
        munmap(raw_bitmap, map_size);
      exit(1);
    }
  }

  // printf("Number of threads: %d\n", omp_get_max_threads());
  // printf("Number of NUMA nodes: %d\n", numa_max_node()+1);
  // printf("Threads per NUMA node: %d\n", threads_per_numa);
  // printf("Total Size in bytes: %d\n", map_size);

  // Group the numa
  std::tuple<size_t, size_t, size_t> numa_params = {threads_per_numa, bitmap_size, region_size};
  for (ssize_t x = 1; x < std::log2(G.num_nodes()); ++x) {
    // Equation 9
    ssize_t thetaPrime = ThetaPrime(x, epsilonPrime, l, k, G.num_nodes(),
                                    std::forward<execution_tag>(ex_tag));
    size_t delta = thetaPrime - RR.size();
    record.ThetaPrimeDeltas.push_back(delta);

    auto timeRRRSets = measure<>::exec_time([&]() {

      auto start_gen = std::chrono::high_resolution_clock::now();
      //RR.insert(RR.end(), delta, RRRset<GraphTy>(allocator)); // original implementation
      RR.resize(RR.size() + delta); // call default constructor

      auto begin = RR.end() - delta;

      GenerateRRRSets(G, counter, mapped_bitmap, numa_params, generator, begin, RR.end(), record,
                      std::forward<diff_model_tag>(model_tag),
                      std::forward<execution_tag>(ex_tag));
                    
      auto end_gen = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_gen - start_gen).count();
      std::cout << "Time spent in sampling of generating RRRsets: " << duration << std::endl;
    });
    record.ThetaEstimationGenerateRRR.push_back(timeRRRSets);
    
    double f;

    auto timeMostInfluential = measure<>::exec_time([&]() {
      auto start_atomicAdd = std::chrono::high_resolution_clock::now();
      const auto &S =
          FindMostInfluentialSet_atomics(G, counter, RR.size(), k, RR);
      // const auto &S =
      //     FindMostInfluentialSet(G, CFG, RR, record, generator.isGpuEnabled(),
      //                            std::forward<execution_tag>(ex_tag));
      f = S.first;
      
      auto end_atomicAdd = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_atomicAdd - start_atomicAdd).count();
      std::cout << "Time spent in sampling of seed selection: " << duration << std::endl;
    });
    // Stop counting the events in the Event Set
    // retval = PAPI_stop(event_set, values);
    // if (retval != PAPI_OK) {
    //     handle_error(retval);
    // }
    
    record.ThetaEstimationMostInfluential.push_back(timeMostInfluential);
    if (f >= std::pow(2, -x)) {
      // std::cout << "Fraction " << f << std::endl;
      LB = (G.num_nodes() * f) / (1 + epsilonPrime);
      break;
    }
  }

  
  size_t theta = Theta(epsilon, l, k, LB, G.num_nodes());

  auto end = std::chrono::high_resolution_clock::now();

  record.ThetaEstimationTotal = end - start;

  record.Theta = theta;
  spdlog::get("console")->info("Theta {}", theta);

  record.GenerateRRRSets = measure<>::exec_time([&]() {
    auto start_gen = std::chrono::high_resolution_clock::now();
    if (theta > RR.size()) {
      size_t final_delta = theta - RR.size();
      // RR.insert(RR.end(), final_delta, RRRset<GraphTy>(allocator)); // original implementation
      RR.resize(RR.size() + final_delta); // call default constructor

      auto begin = RR.end() - final_delta;

      GenerateRRRSets(G, counter, mapped_bitmap, numa_params, generator, begin, RR.end(), record,
                      std::forward<diff_model_tag>(model_tag),
                      std::forward<execution_tag>(ex_tag));
    }
    auto end_gen = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_gen - start_gen).count();
    std::cout << "Time spent in sampling of generating RRRsets: " << duration << std::endl;
  });
  return RR;
}

//! The IMM algroithm for Influence Maximization
//!
//! \tparam GraphTy The type of the input graph.
//! \tparam ConfTy The configuration type
//! \tparam PRNG The type of the parallel random number generator.
//! \tparam diff_model_tag Type-Tag to selecte the diffusion model.
//! \tparam execution_tag Type-Tag to select the execution policy.
//!
//! \param G The input graph.  The graph is transoposed.
//! \param CFG The configuration.
//! \param l Parameter usually set to 1.
//! \param gen The parallel random number generator.
//! \param model_tag The diffusion model tag.
//! \param ex_tag The execution policy tag.
template <typename GraphTy, typename ConfTy, typename GeneratorTy,
          typename diff_model_tag>
auto IMM(const GraphTy &G, const ConfTy &CFG, double l, GeneratorTy &gen,
         diff_model_tag &&model_tag, omp_parallel_tag &&ex_tag) {
  using vertex_type = typename GraphTy::vertex_type;
  size_t k = CFG.k;
  double epsilon = CFG.epsilon;
  auto &record(gen.execution_record());

  l = l * (1 + 1 / std::log2(G.num_nodes()));

  // Counter initialized for kernel fusion.
  long *counter = new long[G.num_nodes()];

  //Initialize the counter
  #pragma omp parallel
    {
        size_t n = G.num_nodes();
        long t = omp_get_thread_num();
        long p = omp_get_max_threads();
        long vstrt = t * n / p;
        long vend = (t + 1) * n / p;
        long vcount = vend - vstrt;
        // Initialzie the global counter in parallel
        for(auto i = vstrt; i < vend; ++i){
            counter[i] = 0;
        }
    }


  auto RR =
      Sampling(G, counter, CFG, l, gen, record, std::forward<diff_model_tag>(model_tag),
               std::forward<omp_parallel_tag>(ex_tag));


  auto start = std::chrono::high_resolution_clock::now();

  auto start_atomicAdd = std::chrono::high_resolution_clock::now();
  
  // RRRset partitioning
  const auto &S =
  FindMostInfluentialSet_atomics(G, counter, RR.size(), k, RR);


  

  // Original Implementation

  // const auto &S =
  // FindMostInfluentialSet(G, CFG, R, record, gen.isGpuEnabled(),
  //                        std::forward<omp_parallel_tag>(ex_tag));  

  
  auto end_atomicAdd = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_atomicAdd - start_atomicAdd).count();
  std::cout << "Time spent in finding the final seed set: " << duration << std::endl;

  auto end = std::chrono::high_resolution_clock::now(); 
  record.FindMostInfluentialSet = end - start;

  start = std::chrono::high_resolution_clock::now();
  size_t total_size = 0; // unit: bit
#pragma omp parallel for reduction(+:total_size)
  for (size_t i = 0; i < RR.size(); ++i) {
    if (auto p = std::get_if<std::unique_ptr<std::vector<uint64_t>>>(&RR[i])) {
      // uint64_t vectors, each element is 64 bits
      total_size += (*p)->size() * 64;
    }
    else if (auto p = std::get_if<std::unique_ptr<std::vector<bool>>>(&RR[i])) {
      // boolean vectors are specialized such that each element is 1 bit
      total_size += (*p)->size();
    }
  }
  total_size /= 8; // convert to bytes

  std::free(counter);
  record.RRRSetSize = total_size;
  end = std::chrono::high_resolution_clock::now();
  record.Total = end - start;

  return S.second;
}

}  // namespace ripples

#endif  // RIPPLES_IMM_H
