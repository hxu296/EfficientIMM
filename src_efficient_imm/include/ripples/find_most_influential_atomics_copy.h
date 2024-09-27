#ifndef RIPPLES_FIND_MOST_INFLUENTIAL_ATOMICS_H
#define RIPPLES_FIND_MOST_INFLUENTIAL_ATOMICS_H

#include <algorithm>
#include <cassert>
#include <chrono>
#include <functional>
#include <iostream>
#include <omp.h>
#include <string.h>
#include <typeinfo>
#include <atomic>
#include <unordered_map>
#include "ripples/utility.h"
#include "ripples/graph.h"
#include "ripples/streaming_rrr_generator.h"

// #include "/global/homes/h/hanwu313/boost_1_82_0/boost/dynamic_bitset.hpp"
//! \brief Select k seeds starting from the a list of Random Reverse
//! Reachability Sets.
//!
//! \tparam RRRset The type storing Random Reverse Reachability Sets.
//!
//! \param G The input graph.
//! \param k The size of the seed set.
//! \param RRRsets A vector of Random Reverse Reachability sets.
//!
//! \return a pair where the size_t is the number of RRRset covered and
//! the set of vertices selected as seeds.

// The SeedPair Struct to be stored in the returned most influential set
namespace ripples {

template<typename vertex_type>
struct SeedPair {
    vertex_type vertex_id;
    long inf_count;
};
// static_assert(std::is_standard_layout_v<SeedPair>);

// Return the max count vertex with its id and the count number in the assigned vertex range
template<typename vertex_type>
void static reduceMaxCountVertex(SeedPair<vertex_type>* glob_pairs,
                                 const long &t, const long &p,
                                //  std::atomic<long> *counter,
                                 long* counter, 
                                 const long &vstrt, const long &vcount, const long &vend,
                                 SeedPair<vertex_type> &selected_seedpair) {                               
    long max_val = 0;
    vertex_type max_idx = 0;
    // Copy to local
    auto loc_counter = &counter[vstrt]; 
    for (auto i = vstrt; i < vend; ++i) {
        auto cntr_pos = i - vstrt;
        if (loc_counter[cntr_pos] > max_val) {
            max_val = loc_counter[cntr_pos];
            max_idx = i;
        }
    }
    // SeedPair<vertex_type> ret_pair = {max_idx, max_val};
    glob_pairs[t] = {max_idx, max_val};

#pragma omp barrier

// More optimization can be done through all reduce on the glob pairs
// by comparing the cnts and the corresponding vertex id
#pragma omp single
    {
        for (auto i = 0l; i < p; ++i) {
            if (glob_pairs[i].inf_count > selected_seedpair.inf_count) {
                selected_seedpair = glob_pairs[i];
            }
        }
    }
}
// Binary Search
// template <typename RRRset>
// bool binarySearch(const RRRset &sorted_list, long size,
//                          long target) {
//     int left = 0;
//     int right = size - 1;

//     while (left <= right) {
//         int mid = left + (right - left) / 2;
//         if (sorted_list[mid] == target) {
//             // Element found
//             return true;
//         } else if (sorted_list[mid] < target) {
//             // Search in the right half
//             left = mid + 1;
//         } else {
//             // Search in the left half
//             right = mid - 1;
//         }
//     }
//     // Element not found
//     return false;
// }

template<typename LP, typename T, typename BP>
void static dma_bcast1_add(LP sgl, T val, std::size_t n, BP base) {
    #pragma unroll
    for (auto i = 0ul; i < n; ++i){
        if(sgl[i]){
            #pragma omp atomic 
            base[i] = base[i] + val;
        }
    }
}

// TIGRE-implemented find most influential set
template <typename GraphTy>
auto FindMostInfluentialSet_atomics(const GraphTy &G, long* counter, long theta, long k,
                            const std::vector<boost::dynamic_bitset<>>& ref_RRRsets) {
    using vertex_type = typename GraphTy::vertex_type;

    std::cout<<"atomics bitmap version"<<std::endl;

    long n = G.num_nodes();
    // Allocate global counter array
    long* cur_counter = new long[n];

    // To avoid the edge case of number of threads bigger than the vertices
    long p = omp_get_max_threads();
    p = std::min(theta, p);
    p = std::min(p, n);

    SeedPair<vertex_type> selected_seedpair = {0, -1};
    std::vector<vertex_type> InfMaxSet(k);
    vertex_type max_vert = 0;
    SeedPair<vertex_type> *glob_pairs = new SeedPair<vertex_type>[p]; // vertex, count pair for each thread
    size_t uncovered = ref_RRRsets.size(); // The number of uncovered RRRsets
    bool *loc_covered = new bool[theta]; // Track the coverage of RRRsets for each thread
    bool update_inc = false;
    long num_seeds = 0; // Track the number of seeds to return

#pragma omp parallel num_threads(p) 
    {
        long t = omp_get_thread_num();

        // Start of the RRRsets splitting and histogramming-like counters update.
        // Number of RRRsets for each thread
        long theta_strt = t * theta / p;
        long theta_end = (t + 1) * theta / p;
        long num_RRRsets = theta_end - theta_strt;

        long vstrt = t * n / p;
        long vend = (t + 1) * n / p;
        long vcount = vend - vstrt;

        // Copy the global counter in parallel
        for(auto i = vstrt; i < vend; ++i){
            cur_counter[i] = counter[i];
        }
        // Initialize the local covered array
        for (int i = theta_strt; i < theta_end; ++i) {
            loc_covered[i] = false;
        }

        // auto start_atomicAdd = std::chrono::high_resolution_clock::now();
        while (num_seeds < k) {

        // Explicit barrier to ensure the completion of all updates
#pragma omp barrier
            // Find the selected pair of maximum cur_counter and the corresponding index
            // auto start_atomicAdd = std::chrono::high_resolution_clock::now();
            reduceMaxCountVertex(glob_pairs, t, p, cur_counter, vstrt, vcount, vend, selected_seedpair);
#pragma omp single
            {
                // Update the InfMaxSet to return
                InfMaxSet[num_seeds] = selected_seedpair.vertex_id;
                max_vert = selected_seedpair.vertex_id;
                uncovered -= selected_seedpair.inf_count;
                num_seeds++;
                // Maybe need a better way to determine the condition
                update_inc = (uncovered < selected_seedpair.inf_count) ? true : false;
                // std::cout << "vert: " << max_vert << " cnt: " << selected_seedpair.inf_count << std::endl;
                selected_seedpair = {0, -1};
            }
            if (num_seeds == k)
                break;
            // Increment the number of seeds and break out the loop if all the seeds are found

            // Timer for the whole updates counter
            auto start_atomicAdd = std::chrono::high_resolution_clock::now();
            if(update_inc) {
                for(auto i = vstrt; i < vend; ++i){
                    cur_counter[i] = 0;
                }
                for (int it_theta = theta_strt; it_theta < theta_end; ++it_theta) {
                    if (loc_covered[it_theta] == true)
                        continue;
                    // get the RRRset size
                    long cur_set_size = ref_RRRsets[it_theta].size();
                    // Do the sorted binary search on each sorted list to increment the counter
                    if (!ref_RRRsets[it_theta][max_vert]) {
                        dma_bcast1_add(ref_RRRsets[it_theta], 1l, cur_set_size, cur_counter);
                    }else{
                        loc_covered[it_theta] = true;
                    }
                }
            }else {
                for (int it_theta = theta_strt; it_theta < theta_end; ++it_theta) {
                    if (loc_covered[it_theta] == true)
                        continue;
                    // get the RRRset size
                    long cur_set_size = ref_RRRsets[it_theta].size();
                    // Do the sorted binary search on each sorted list to decrement the counter
                    if (ref_RRRsets[it_theta][max_vert]) {
                        dma_bcast1_add(ref_RRRsets[it_theta], -1l, cur_set_size, cur_counter);
                        loc_covered[it_theta] = true;
                    }
                }
            }
            // auto end_atomicAdd = std::chrono::high_resolution_clock::now();
            // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_atomicAdd - start_atomicAdd);
            // time_atomics[t] += duration.count();
        }
        // std::free(loc_covered);
    }
    std::free(loc_covered);
    double f = (double) (ref_RRRsets.size() - uncovered) / ref_RRRsets.size();
    std::free(cur_counter);

    // Collect the timer
    // double max_atomic_time = 0.0;
    // for(auto & elem : time_atomics) {
    //     std::cout << elem << " ";
    //     max_atomic_time = (elem > max_atomic_time) ? elem : max_atomic_time;
    // }
    // std::cout << std::endl;
    // std::cout << "Time spent atomics (inside): " << max_atomic_time << std::endl;  
    return std::make_pair(f, InfMaxSet);
}
}

#endif