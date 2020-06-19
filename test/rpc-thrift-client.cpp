#include "../include/JCache.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <fstream>
#include <random>
#include <thread>
#include <vector>

int MAX_PUT_NUM = 100;

using std::string;
using std::vector;

vector<std::thread> threads;
size_t thread_num = 1;
string filelistname;
string server_addr = "127.0.0.1";
int server_port = 8080;
vector<vector<size_t>> rep_times;
string root_dir;

vector<string> filenames;
vector<size_t> random_idx;
vector<std::shared_ptr<std::atomic<size_t>>> finished_req;
vector<std::shared_ptr<std::atomic<size_t>>> finished_traffic;
vector<std::shared_ptr<std::atomic<size_t>>> finished_time;

std::thread log_thread;

std::atomic<ssize_t> time_left;
size_t REQNUM = 100000;

int mode = 0;
bool start_log;

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
    vector<string> local_filenames = filenames;

    size_t idx = tid * filenames.size();
    for (size_t req = 0;; req++) {
        if (start_log) {
            if (time_left < 0) {
                break;
            }
        } else {
            if (req >= REQNUM) {
                break;
            }
        }
        auto fname = filenames[random_idx[idx]];
        // printf("get %s\n", fname.c_str());
        auto t1 = get_wall_time();
        if (mode == 0) {
            ret_buff = client.get_serialized_header_random_crop(fname);
        } else if (mode == 1) {
            ret_buff = client.get_serialized_raw_file(fname);
        } else if (mode == 2) {
            ret_buff = client.get_serialized_header_ROI(fname, 0, 0, 224, 224);
        }
        auto t2 = get_wall_time();
        rep_times[tid].push_back(t2 - t1);
        auto transfered = finished_traffic[tid]->fetch_add(ret_buff.size());
        auto total_req = finished_req[tid]->fetch_add(1);
        auto time = finished_time[tid]->fetch_add(t2 - t1);
        if (tid == 0 && req % 100 == 0) {
            printf("req: %ld, ave_latency: %lf, transfered: %ld\n", req, (double) time / total_req, transfered);
        }
        idx++;
        if (idx >= random_idx.size()) {
            idx = 0;
        }
    }
}

void log_thread_func() {
    int t = 0;
    for (;;) {
        if (time_left < 0) {
            break;
        }
        size_t total_req = 0;
        size_t total_traffic = 0;
        size_t total_rep_time = 0;
        for (int i = 0; i < thread_num; i++) {
            total_req += finished_req[i]->load();
            total_traffic += finished_traffic[i]->load();
            total_rep_time += finished_time[i]->load();
        }
        printf("time %d, req %ld, traffic %ld, latency %lf\n", t, total_req, total_traffic,
               total_rep_time * 1.0 / total_req);
        t++;
        std::this_thread::sleep_for(std::chrono::seconds(1));
        time_left--;
    }
}

int main(int argc, char **argv) {
    assert(argc == 10);

    server_addr = string(argv[1]);
    server_port = std::atol(argv[2]);
    filelistname = string(argv[3]);
    root_dir = string(argv[4]);
    size_t total_time = std::atoll(argv[5]);
    thread_num = std::atoll(argv[6]);
    mode = std::atoi(argv[7]);
    MAX_PUT_NUM = std::atoi(argv[8]);
    string enable_write = argv[9];

    printf("server addr %s, port %d\n", server_addr.c_str(), server_port);
    printf("filelist %s\n", filelistname.c_str());
    printf("root dir %s\n", root_dir.c_str());
    printf("run time is %ld\n", total_time);
    printf("thread num is %ld\n", thread_num);
    printf("mode %d\n", mode);
    printf("put to server %d\n", MAX_PUT_NUM);

    assert(mode == 0 || mode == 1 || mode == 2);

    rep_times.resize(thread_num);
    for (int i = 0; i < thread_num; i++) {
        finished_req.push_back(std::make_shared<std::atomic<size_t>>(0));
        finished_traffic.push_back(std::make_shared<std::atomic<size_t>>(0));
        finished_time.push_back(std::make_shared<std::atomic<size_t>>(0));
    }
    time_left = total_time;
    if (time_left > 200) {
        REQNUM = time_left;
        start_log = false;
    } else {
        start_log = true;
    }

    if (enable_write == "true") {
        // read file list
        {
            int count = 0;
            int fail = 0;
            jcache::JPEGCacheClient client(server_addr, server_port);
            std::ifstream infile(filelistname);
            assert(infile.good());
            std::string line;

            std::ofstream ofile("list.txt", std::ios::trunc);

            while (std::getline(infile, line)) {
                if (line.size() > 0) {
                    string fname = root_dir + "/" + line;
                    // printf("puting %s\n", fname.c_str());
                    auto ret = client.put(fname);
                    if (ret == 0) {
                        // printf("ok\n");
                        filenames.push_back(fname);
                        ofile << fname << std::endl;
                        count++;
                        if (count >= MAX_PUT_NUM) {
                            break;
                        }
                    } else {
                        fail++;
                    }
                }
            }
            printf("fail %d failes\n", fail);
        }
    } else {
        std::ifstream infile("list.txt");
        assert(infile.good());
        std::string line;
        while (std::getline(infile, line)) {
            if (line.size() > 0) {
                filenames.push_back(line);
            }
        }
    }
    printf("total %ld files\n", filenames.size());
    for (int i = 0; i < filenames.size() * thread_num; i++) {
        random_idx.push_back(i % filenames.size());
    }

    {
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(random_idx.begin(), random_idx.end(), g);
    }

    if (start_log) {
        log_thread = std::thread(log_thread_func);
    }

    int64_t start_time = get_wall_time();
    for (size_t i = 0; i < thread_num; i++) {
        threads.push_back(std::thread(client_func, i));
    }
    for (size_t i = 0; i < thread_num; i++) {
        threads[i].join();
    }
    int64_t end_time = get_wall_time();

    if (start_log) {
        log_thread.join();
    }

    double total_time_in_ms = (end_time - start_time) / 1000.0;

    size_t total_req = 0;
    size_t total_transfered = 0;
    for (int i = 0; i < thread_num; i++) {
        total_transfered += finished_traffic[i]->load();
        total_req += finished_req[i]->load();
    }
    printf("Requests: %ld, Total time: %.3lfms, qps: %.3f\n", total_req, total_time_in_ms,
           ((double) total_req / (total_time_in_ms / 1000.0)));
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