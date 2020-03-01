#include "int2bytes.h"
#include <cstring>
void uint16_to_bytes(uint16_t val, unsigned char *bytes) {
    unsigned char *b = (unsigned char *) bytes;
    unsigned char *p = (unsigned char *) &val;
    memcpy(b, p, sizeof(uint16_t));
}

uint16_t bytes_to_uint16(unsigned char *bytes) {
    uint16_t res;
    memcpy(&res, bytes, sizeof(uint16_t));
    return res;
}

void uint32_to_bytes(uint32_t val, unsigned char *bytes) {
    unsigned char *b = (unsigned char *) bytes;
    unsigned char *p = (unsigned char *) &val;
    memcpy(b, p, sizeof(uint32_t));
}

uint32_t bytes_to_uint32(unsigned char *bytes) {
    uint32_t res;
    memcpy(&res, bytes, sizeof(uint32_t));
    return res;
}

void uint64_to_bytes(uint64_t val, unsigned char *bytes) {
    unsigned char *b = (unsigned char *) bytes;
    unsigned char *p = (unsigned char *) &val;
    memcpy(b, p, sizeof(uint64_t));
}

uint64_t bytes_to_uint64(unsigned char *bytes) {
    uint64_t res;
    memcpy(&res, bytes, sizeof(uint64_t));
    return res;
}
