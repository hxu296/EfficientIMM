#include <unistd.h>
#include <sys/mman.h>

#ifndef BITMAP_H
#define BITMAP_H

namespace ripples{
    #define BYTE_TO_BIT 8
    namespace mybitmap{

      template <typename data_type>
      void inline set_bit(data_type* bitmap, size_t bit_index) {
          size_t bits_per_entry = sizeof(data_type) * BYTE_TO_BIT;
          size_t entry_index = bit_index / bits_per_entry; 
          size_t bit_pos = bit_index % bits_per_entry; 
          // Set the bit at the position to 1
          bitmap[entry_index] |= (static_cast<data_type>(1) << bit_pos);
      }

      template <typename data_type>
      void inline reset_bit(data_type* bitmap, size_t bit_index) {
          size_t bits_per_entry = sizeof(data_type) * BYTE_TO_BIT;
          size_t entry_index = bit_index / bits_per_entry; 
          size_t bit_pos = bit_index %  bits_per_entry; 
          // reset the bit at the position to 0
          bitmap[entry_index] &= ~(static_cast<data_type>(1) << bit_pos);
      }

      template <typename data_type>
      void inline reset_bitmap(data_type* bitmap, size_t num_bits) {
          // Calculate the number of data_type elements in byte
          size_t bits_per_entry = sizeof(data_type) * BYTE_TO_BIT;
          size_t bitmap_size = (num_bits + bits_per_entry - 1) / bits_per_entry;
          // for (data_type i = 0; i < bitmap_size; ++i) {
          //     // Set each datatype in the array to 0
          //     bitmap[i] &= static_cast<data_type>(0); 
          // }
          // Readjust the bitmap_size to byte
          bitmap_size *= sizeof(size_t);
          memset(bitmap, 0, bitmap_size);
      }

      template <typename data_type>
      bool inline check_bit_set(data_type* bitmap, size_t bit_index) {
          size_t bits_per_entry = sizeof(data_type) * BYTE_TO_BIT;
          size_t entry_index = bit_index / bits_per_entry;  
          size_t bit_pos = bit_index % bits_per_entry;
          // Check
          return (bitmap[entry_index] & (static_cast<data_type>(1) << bit_pos)) != 0; 
      }

      template <typename data_type>
      bool inline check_bit_unset(data_type* bitmap, size_t bit_index) {
          size_t bits_per_entry = sizeof(data_type) * BYTE_TO_BIT;
          size_t entry_index = bit_index / bits_per_entry;  
          size_t bit_pos = bit_index % bits_per_entry;
          // Check
          return (bitmap[entry_index] & (static_cast<data_type>(1) << bit_pos)) == 0; 
      }
    }


}

#endif