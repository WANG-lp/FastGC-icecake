#pragma once
#include <cstdint>

// data always stored in SYSTEM ENDIAN format

void uint16_to_bytes(uint16_t val, unsigned char *bytes);

uint16_t bytes_to_uint16(unsigned char *bytes);

void uint32_to_bytes(uint32_t val, unsigned char *bytes);

uint32_t bytes_to_uint32(unsigned char *bytes);

void uint64_to_bytes(uint64_t val, unsigned char *bytes);

uint64_t bytes_to_uint64(unsigned char *bytes);
