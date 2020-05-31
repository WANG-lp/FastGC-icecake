#pragma once
#include "../libgpujpeg/gpujpeg_decoder.h"
#include "gpujpeg_decoder_internal.h"
int init_decoder_with_image(struct gpujpeg_decoder* decoder, const char* image_path);