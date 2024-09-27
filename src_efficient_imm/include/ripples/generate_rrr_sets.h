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

#ifndef RIPPLES_GENERATE_RRR_SETS_H
#define RIPPLES_GENERATE_RRR_SETS_H

#include <algorithm>
#include <queue>
#include <utility>
#include <vector>
#include <iostream>
#include "omp.h"

#include "ripples/diffusion_simulation.h"
#include "ripples/graph.h"
#include "ripples/imm_execution_record.h"
#include "ripples/utility.h"
#include "ripples/streaming_rrr_generator.h"

#include "trng/uniform01_dist.hpp"
#include "trng/uniform_int_dist.hpp"

#include "ripples/bitmap.h"
#include <variant> 
#include "ripples/bitmap.h"

#ifdef ENABLE_MEMKIND
#include "memkind_allocator.h"
#include "pmem_allocator.h"
#endif

#ifdef ENABLE_METALL_RRRSETS
#include "metall/metall.hpp"
#include "metall/container/vector.hpp"
#endif

namespace ripples {

#if defined ENABLE_MEMKIND
template<typename vertex_type>
using RRRsetAllocator = libmemkind::pmem::allocator<vertex_type>;
#elif defined ENABLE_METALL_RRRSETS
template<typename vertex_type>
using RRRsetAllocator = metall::manager::allocator_type<vertex_type>;

metall::manager &metall_manager_instance(std::string path) {
  static metall::manager manager(metall::create_only, path.c_str());
  return manager;
}

#else
template <typename vertex_type>
using RRRsetAllocator = std::allocator<vertex_type>;
using AdaptiveRRRType = std::variant<std::unique_ptr<std::vector<uint64_t>>, std::unique_ptr<std::vector<bool>>>;
#endif

// //! \brief The Random Reverse Reachability Sets type
// template <typename GraphTy>
// using RRRset =
// #ifdef  ENABLE_METALL_RRRSETS
//     metall::container::vector<typename GraphTy::vertex_type,
//                               RRRsetAllocator<typename GraphTy::vertex_type>>;
// #else
//     std::vector<typename GraphTy::vertex_type,
//                               RRRsetAllocator<typename GraphTy::vertex_type>>;
// #endif
// template <typename GraphTy>
// using RRRsets = std::vector<RRRset<GraphTy>>;

//! \brief Execute a randomize BFS to generate a Random RR Set.
//!
//! \tparam GraphTy The type of the graph.
//! \tparam PRNGGeneratorTy The type of pseudo the random number generator.
//! \tparam diff_model_tag The policy for the diffusion model.
//!
//! \param G The graph instance.
//! \param r The starting point for the exploration.
//! \param generator The pseudo random number generator.
//! \param result The RRR set
//! \param tag The diffusion model tag.
template <typename GraphTy, typename PRNGeneratorTy, typename diff_model_tag>
AdaptiveRRRType AddRRRSet(long* counter, size_t* thread_bitmap, const GraphTy &G, typename GraphTy::vertex_type r,
               PRNGeneratorTy &generator, diff_model_tag &&tag) {
  using vertex_type = typename GraphTy::vertex_type;

  std::vector<bool> visited(G.num_nodes(), false);
  const size_t rrrset_size_threshold = G.num_nodes() * 0.016;
  std::vector<uint64_t> dense_rrrset;
  trng::uniform01_dist<float> value;

  std::queue<vertex_type> queue;
  // allocate and initialize visited
  ripples::mybitmap::reset_bitmap(thread_bitmap, G.num_nodes());
  queue.push(r);
  visited[r] = true;
  ripples::mybitmap::set_bit(thread_bitmap, r);
  dense_rrrset.push_back(r);

  while (!queue.empty()) {
    vertex_type v = queue.front();
    queue.pop();

    if (std::is_same<diff_model_tag, ripples::independent_cascade_tag>::value) {
      for (auto u : G.neighbors(v)) {
        if (ripples::mybitmap::check_bit_unset(thread_bitmap, u.vertex) && value(generator) <= u.weight) {
          queue.push(u.vertex);
          visited[u.vertex] = true;
          ripples::mybitmap::set_bit(thread_bitmap, u.vertex);
          if(dense_rrrset.size() < rrrset_size_threshold) dense_rrrset.push_back(u.vertex);
        }
      }
    } else if (std::is_same<diff_model_tag,
                            ripples::linear_threshold_tag>::value) {
      float threshold = value(generator);
      for (auto u : G.neighbors(v)) {
        threshold -= u.weight;

        if (threshold > 0) continue;

        if (ripples::mybitmap::check_bit_unset(thread_bitmap, u.vertex)) {
          queue.push(u.vertex);
          visited[u.vertex] = true;
          ripples::mybitmap::set_bit(thread_bitmap, u.vertex);
          if(dense_rrrset.size() < rrrset_size_threshold) dense_rrrset.push_back(u.vertex);
        }
        break;
      }
    } else {
      throw;
    }
  }


  if(dense_rrrset.size() < rrrset_size_threshold) {
    std::stable_sort(dense_rrrset.begin(), dense_rrrset.end());
    for(auto & node : dense_rrrset) {
      #pragma omp atomic
        counter[node] += 1;
    }
    return std::make_unique<std::vector<uint64_t>>(std::move(dense_rrrset));
  } else {
    for(int i = 0; i < G.num_nodes(); i++) {
        if(visited[i]) {
          #pragma omp atomic
            counter[i] += 1;
        }
    }
    return std::make_unique<std::vector<bool>>(std::move(visited));
  }
}

// //! \brief Generate Random Reverse Reachability Sets - sequential.
// //!
// //! \tparam GraphTy The type of the garph.
// //! \tparam PRNGeneratorty The type of the random number generator.
// //! \tparam ItrTy A random access iterator type.
// //! \tparam ExecRecordTy The type of the execution record
// //! \tparam diff_model_tag The policy for the diffusion model.
// //!
// //! \param G The original graph.
// //! \param generator The random numeber generator.
// //! \param begin The start of the sequence where to store RRR sets.
// //! \param end The end of the sequence where to store RRR sets.
// //! \param model_tag The diffusion model tag.
// //! \param ex_tag The execution policy tag.
// template <typename GraphTy, typename PRNGeneratorTy,
//           typename ItrTy, typename ExecRecordTy,
//           typename diff_model_tag>
// void GenerateRRRSets(GraphTy &G, PRNGeneratorTy &generator,
//                      ItrTy begin, ItrTy end,
//                      ExecRecordTy &,
//                      diff_model_tag &&model_tag,
//                      sequential_tag &&ex_tag) {
//   trng::uniform_int_dist start(0, G.num_nodes());
  
//   //Just a placeholder to call the changed AddRRRset code
//   long *counter = new long[G.num_nodes()];

//   for (auto itr = begin; itr < end; ++itr) {
//     typename GraphTy::vertex_type r = start(generator[0]);
//     AddRRRSet(counter, G, r, generator[0], *itr,
//               std::forward<diff_model_tag>(model_tag));
//   }
// }

//! \brief Generate Random Reverse Reachability Sets - CUDA.
//!
//! \tparam GraphTy The type of the garph.
//! \tparam PRNGeneratorty The type of the random number generator.
//! \tparam ItrTy A random access iterator type.
//! \tparam ExecRecordTy The type of the execution record
//! \tparam diff_model_tag The policy for the diffusion model.
//!
//! \param G The original graph.
//! \param generator The random numeber generator.
//! \param begin The start of the sequence where to store RRR sets.
//! \param end The end of the sequence where to store RRR sets.
//! \param model_tag The diffusion model tag.
//! \param ex_tag The execution policy tag.
template <typename GraphTy, typename PRNGeneratorTy,
          typename ItrTy, typename ExecRecordTy,
          typename diff_model_tag>
void GenerateRRRSets(const GraphTy &G, long* counter, size_t* mapped_bitmap, std::tuple<size_t, size_t, size_t> numa_params,
                     StreamingRRRGenerator<GraphTy, PRNGeneratorTy, ItrTy, diff_model_tag> &se,
                     ItrTy begin, ItrTy end,
                     ExecRecordTy &,
                     diff_model_tag &&,
                     omp_parallel_tag &&) {
  se.generate(counter, mapped_bitmap, numa_params, begin, end);
}

}  // namespace ripples

#endif  // RIPPLES_GENERATE_RRR_SETS_H
