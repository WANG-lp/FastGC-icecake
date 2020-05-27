#include "../include/JCache.hpp"

int main(int argc, char** argv) {
    jcache::JCache cache(8090);
    cache.serve();
    return 0;
}