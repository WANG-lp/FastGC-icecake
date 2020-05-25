#include "../include/JCache.hpp"

int main(int argc, char** argv) {
    jcache::JCache cache;
    cache.putJPEG(argv[1]);
    return 0;
}