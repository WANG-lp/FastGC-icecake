#include <spdlog/spdlog.h>
#include <sys/time.h>
#include <time.h>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

using std::string;
using std::vector;

int64_t get_wall_time() {
    struct timeval time;
    if (gettimeofday(&time, NULL)) {
        //  Handle error
        return 0;
    }
    return time.tv_sec * 1000000 + time.tv_usec;
}

vector<vector<uchar>> test_file_read(const string& root_dir, const vector<string>& fnames, size_t bench_num) {
    size_t bench_images_num = bench_num;

    vector<vector<uchar>> result;
    result.resize(bench_num);

    size_t total_file_size = 0;
    char last_byte = 0;
    auto start_t = get_wall_time();
#pragma omp parallel for reduction(+ : total_file_size) num_threads(16)
    for (int i = 0; i < bench_images_num; i++) {
        string fullname = root_dir + fnames[i];
        std::ifstream is(fullname, std::ifstream::binary);
        // get length of file:
        is.seekg(0, is.end);
        int length = is.tellg();
        is.seekg(0, is.beg);
        total_file_size += length;
        result[i].resize(length);
        is.read((char*) result[i].data(), length);
        last_byte = last_byte | result[i][length - 1];
    }
    auto end_t = get_wall_time();
    double time = (end_t - start_t) / 1000000.0f;

    // spdlog::info("benched {} files", bench_images_num);
    spdlog::info("==========file read==========");
    spdlog::info("total_size: {}, last byte: {}", total_file_size, uint8_t(last_byte));
    spdlog::info("time: {}s", time);
    double time_per_image = time / bench_images_num;
    spdlog::info("time per file: {}", time_per_image);
    spdlog::info("speed: {} file/second", bench_images_num / time);
    return result;
}

vector<cv::Mat> test_opencv_crop(const vector<cv::Mat>& images_mat, size_t bench_num) {
    size_t bench_images_num = bench_num;

    vector<cv::Mat> result;
    result.resize(bench_num);

    size_t t_height = 0;
    size_t t_width = 0;
    auto start_t = get_wall_time();
#pragma omp parallel for reduction(+ : t_height) reduction(+ : t_width) num_threads(16)
    for (int i = 0; i < bench_images_num; i++) {
        cv::Mat img = images_mat[i];
        cv::Rect roi;
        roi.x = 10;
        roi.y = 10;
        roi.width = 224;
        roi.height = 224;
        if (roi.x + roi.width > img.cols) {
            roi.width = img.cols - roi.x;
        }
        if (roi.y + roi.height > img.rows) {
            roi.height = img.rows - roi.y;
        }
        /* Crop the original image to the defined ROI */
        cv::Mat crop = img(roi);
        t_height += crop.rows;
        t_width += crop.cols;
        result[i] = crop;
    }
    auto end_t = get_wall_time();
    double time = (end_t - start_t) / 1000000.0f;

    // spdlog::info("benched {} files", bench_images_num);
    spdlog::info("==========opencv crop==========");
    spdlog::info("height: {}, width:{}", t_height, t_width);
    spdlog::info("time: {}s", time);
    double time_per_image = time / bench_images_num;
    spdlog::info("time per file: {}", time_per_image);
    spdlog::info("speed: {} file/second", bench_images_num / time);

    return result;
}

vector<vector<uchar>> test_opencv_to_jpeg(const vector<cv::Mat>& images_mat, size_t bench_num) {
    size_t bench_images_num = bench_num;

    vector<vector<uchar>> result;
    result.resize(bench_num);

    size_t jpeg_size = 0;
    auto start_t = get_wall_time();

#pragma omp parallel for reduction(+ : jpeg_size) num_threads(16)
    for (int i = 0; i < bench_images_num; i++) {
        cv::imencode(".jpeg", images_mat[i], result[i]);
        jpeg_size += result[i].size();
    }
    auto end_t = get_wall_time();
    double time = (end_t - start_t) / 1000000.0f;

    // spdlog::info("benched {} files", bench_images_num);
    spdlog::info("==========opencv to_jpeg==========");
    spdlog::info("compressed jpeg size: {}", jpeg_size);
    spdlog::info("time: {}s", time);
    double time_per_image = time / bench_images_num;
    spdlog::info("time per file: {}", time_per_image);
    spdlog::info("speed: {} file/second", bench_images_num / time);

    return result;
}

vector<cv::Mat> test_opencv_decode(const string& root_dir, const vector<string>& fnames, size_t bench_num) {
    size_t bench_images_num = bench_num;

    vector<cv::Mat> result;
    result.resize(bench_num);

    size_t t_height = 0;
    size_t t_width = 0;
    auto start_t = get_wall_time();
#pragma omp parallel for reduction(+ : t_height) reduction(+ : t_width) num_threads(16)
    for (int i = 0; i < bench_images_num; i++) {
        cv::Mat image;
        string fullname = root_dir + fnames[i];
        image = cv::imread(fullname, cv::ImreadModes::IMREAD_COLOR);
        t_height += image.rows;
        t_width += image.cols;
        result[i] = image;
        // break;
    }
    auto end_t = get_wall_time();
    double time = (end_t - start_t) / 1000000.0f;

    // spdlog::info("benched {} files", bench_images_num);
    spdlog::info("==========opencv decode==========");
    spdlog::info("height: {}, width:{}", t_height, t_width);
    spdlog::info("time: {}s", time);
    double time_per_image = time / bench_images_num;
    spdlog::info("time per image: {}", time_per_image);
    spdlog::info("speed: {} images/second", bench_images_num / time);
    return result;
}

vector<cv::Mat> test_opencv_decode2(const vector<vector<uchar>> img_buffs, size_t bench_num) {
    size_t bench_images_num = bench_num;

    vector<cv::Mat> result;
    result.resize(bench_num);

    size_t t_height = 0;
    size_t t_width = 0;
    auto start_t = get_wall_time();
#pragma omp parallel for reduction(+ : t_height) reduction(+ : t_width) num_threads(16)
    for (int i = 0; i < bench_images_num; i++) {
        result[i] = cv::imdecode(img_buffs[i], cv::ImreadModes::IMREAD_COLOR);
        t_height += result[i].rows;
        t_width += result[i].cols;
        // break;
    }
    auto end_t = get_wall_time();
    double time = (end_t - start_t) / 1000000.0f;

    // spdlog::info("benched {} files", bench_images_num);
    spdlog::info("==========opencv decode from buffer==========");
    spdlog::info("height: {}, width:{}", t_height, t_width);
    spdlog::info("time: {}s", time);
    double time_per_image = time / bench_images_num;
    spdlog::info("time per image: {}", time_per_image);
    spdlog::info("speed: {} images/second", bench_images_num / time);
    return result;
}

int main(int argc, char** argv) {
    char* imagelist = argv[1];
    char* croppedlist = argv[2];

    size_t benched_num = std::atoi(argv[3]);

    vector<string> fnames;
    std::ifstream infile(imagelist);
    std::string line;
    while (std::getline(infile, line)) {
        fnames.push_back(line);
    }
    spdlog::info("total file: {}", fnames.size());
    if (benched_num == 0) {
        benched_num = fnames.size();
    }

    vector<string> fnames_cropped;
    std::ifstream c_infile(croppedlist);
    while (std::getline(c_infile, line)) {
        fnames_cropped.push_back(line);
    }
    spdlog::info("total file: {}", fnames_cropped.size());

    auto file_raw = test_file_read("/mnt/nvme-ssd/lipeng/imagenet/", fnames, benched_num);
    auto decoded_images = test_opencv_decode2(file_raw, benched_num);
    auto croped_imgs = test_opencv_crop(decoded_images, benched_num);
    auto croped_raw = test_opencv_to_jpeg(croped_imgs, benched_num);
    test_opencv_decode2(croped_raw, benched_num);
    return 0;
}