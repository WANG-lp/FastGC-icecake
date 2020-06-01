#pragma once
#include <atomic>
#include <string>
#include <thread>
#include <vector>

using std::string;
using std::vector;
struct gpujpeg_decoder;

namespace jpeg_dec {

class GPUDecoder {
   public:
    GPUDecoder(int thread_num);
    GPUDecoder(int thread_num, const string &init_image);

    ~GPUDecoder();
    int do_decode(void *jpeg_header, uint8_t *out_ptr);
    int do_decode_phase1(size_t which_decode, void *jpeg_header);
    int do_decode_phase2(size_t which_decode, uint8_t *out_ptr);

   private:
    std::atomic<size_t> decoder_idx;
    vector<gpujpeg_decoder *> decoders;
};
}  // namespace jpeg_dec