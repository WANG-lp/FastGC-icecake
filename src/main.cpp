#include "../include/JCache.hpp"

int main(int argc, char** argv) {
    int port = 8090;
    if (argc > 1) {
        port = atoi(argv[1]);
    }
    printf("server listen on port %d\n", port);
    jcache::JCache cache(port);
    cache.serve();
    return 0;
}