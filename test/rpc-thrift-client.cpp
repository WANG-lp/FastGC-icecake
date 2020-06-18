#include "../include/JCache.hpp"

#include <algorithm>
#include <fstream>
#include <thread>
#include <vector>

using std::string;
using std::vector;

vector<std::thread> threads;
size_t thread_num = 1;
string filelistname;
size_t REQNUM = 500000;
string server_addr = "127.0.0.1";
int server_port = 8080;
vector<vector<size_t>> rep_times;
vector<size_t> transfered_data;
string root_dir;

vector<string> filenames;

int64_t get_wall_time() {
    struct timeval time;
    if (gettimeofday(&time, NULL)) {
        //  Handle error
        return 0;
    }
    return time.tv_sec * 1000000 + time.tv_usec;
}

void client_func(int tid) {
    jcache::JPEGCacheClient client(server_addr, server_port);
    string ret_buff;
    for (size_t i = 0; i < REQNUM; i++) {
        auto t1 = get_wall_time();
        ret_buff = client.get_serialized_header_random_crop("/lwangay/test_7687.jpeg");
        // ret_buff = client.get_serialized_raw_file("/lwangay/test_7687.jpeg");
        // ret_buff = client.get_serialized_header_ROI("/lwangay/test_7687.jpeg", 0, 0, 224, 224);

        auto t2 = get_wall_time();
        transfered_data[tid] += ret_buff.size();
        rep_times[tid].push_back(t2 - t1);
    }
}

int main(int argc, char **argv) {
    assert(argc == 7);

    server_addr = string(argv[1]);
    server_port = std::atol(argv[2]);
    filelistname = string(argv[3]);
    root_dir = string(argv[4]);
    REQNUM = std::atoll(argv[5]);
    thread_num = std::atoll(argv[6]);

    printf("server addr %s, port %d\n", server_addr.c_str(), server_port);
    printf("filelist %s\n", filelistname.c_str());
    printf("root dir %s\n", root_dir.c_str());
    printf("req num is %ld\n", REQNUM);
    printf("thread num is %ld\n", thread_num);

    rep_times.resize(thread_num);
    transfered_data.resize(thread_num, 0);

    // read file list
    {
        std::ifstream infile(filelistname);
        assert(infile.good());
        std::string line;
        while (std::getline(infile, line)) {
            if (line.size() > 0) {
                filenames.push_back(root_dir + "/" + line);
            }
        }
    }
    printf("total %d files\n", filenames.size());

    {
        jcache::JPEGCacheClient client(server_addr, server_port);
        client.put("/lwangay/test_7687.jpeg");
    }

    int64_t start_time = get_wall_time();
    for (size_t i = 0; i < thread_num; i++) {
        threads.push_back(std::thread(client_func, i));
    }
    for (size_t i = 0; i < thread_num; i++) {
        threads[i].join();
    }
    int64_t end_time = get_wall_time();

    double total_time_in_ms = (end_time - start_time) / 1000.0;

    REQNUM *= thread_num;
    size_t total_transfered = 0;
    for (int i = 0; i < thread_num; i++) {
        total_transfered += transfered_data[i];
    }
    printf("Requests: %ld, Total time: %.3lfms, qps: %.3f\n", REQNUM, total_time_in_ms,
           ((double) REQNUM / (total_time_in_ms / 1000.0)));
    printf("Transfer bytes: %.3lfMB, Total time: %.3lfms, bandwidth: %.3fMB/s\n", total_transfered / (1024 * 1024.0),
           total_time_in_ms, ((double) (total_transfered / (1024.0 * 1024)) / (total_time_in_ms / 1000.0)));

    vector<size_t> all_times;
    for (size_t i = 0; i < thread_num; i++) {
        for (const auto &t : rep_times[i]) {
            all_times.push_back(t);
        }
    }
    std::sort(all_times.begin(), all_times.end());
    size_t all_ave = 0;
    size_t first_ten_ave = 0;
    size_t latest_ten_ave = 0;
    for (size_t i = 0; i < all_times.size(); i++) {
        if (i < all_times.size() / 100.0 * 10) {  // first 10%
            first_ten_ave += all_times[i];
        }
        if (i >= all_times.size() / 100.0 * 90) {  // last 10%
            latest_ten_ave += all_times[i];
        }
        all_ave += all_times[i];
    }
    first_ten_ave /= (all_times.size()) / 100.0 * 10;
    latest_ten_ave /= (all_times.size()) / 100.0 * 10;
    all_ave /= all_times.size();

    // for (size_t i = 1; i <= 100; i++) {
    //     size_t off = (all_times.size() / 100.0) * i;
    //     if (off >= all_times.size()) off = all_times.size() - 1;
    //     printf("%ld ===> %ld\n", i, all_times[off]);
    // }
    printf("first 10%% ave latency: %ld\n", first_ten_ave);
    printf("last 10%% ave latency: %ld\n", latest_ten_ave);
    printf("all ave latency: %ld\n", all_ave);
    return 0;
}