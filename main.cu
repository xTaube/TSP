#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <random>
#include <atomic>
#include <chrono>
#include <cassert>
#include <fstream>
using namespace std;

#define MAX_X 1000
#define MAX_Y 1000
#define MAX_NO_POINTS MAX_X * MAX_Y
#define MAX_DIST MAX_X*MAX_X + MAX_Y*MAX_Y + 1
#define THREAD_SIZE 512
#define SHMEM_SIZE 512

struct city {
    int posX;
    int posY;
    bool visited;
};

struct times {
    double gpu_time;
    double cpu_time;
    double time_diff;
};


__global__ void calculate_dist(city *cities, double *dist, long long int starting_point, long long n) {
    long long tid = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ city sh_cities[SHMEM_SIZE];
    __shared__ city sh_starting_city[SHMEM_SIZE];
    double temp;
    if (tid < n) {
        if (tid == 0) cities[starting_point].visited = true;
        sh_cities[threadIdx.x] = cities[tid];
        sh_starting_city[threadIdx.x] = cities[starting_point];
        __syncthreads();
        if (!sh_cities[threadIdx.x].visited) {
            double x = (double)sh_cities[threadIdx.x].posX-(double)sh_starting_city[threadIdx.x].posX;
            double y = (double)sh_cities[threadIdx.x].posY-(double)sh_starting_city[threadIdx.x].posY;
            temp = sqrt(x*x + y*y);
            dist[tid] = temp;
            __syncthreads();
        } else dist[tid] = MAX_DIST;
    }
}

__global__ void find_min_reduction(double *dist, double *dist_r, long long int n) {
    __shared__ double sh_dist[SHMEM_SIZE];
    long long int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
        sh_dist[threadIdx.x] = dist[tid];
    else sh_dist[threadIdx.x] = MAX_DIST;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s /= 2) {
        if (threadIdx.x < s && sh_dist[threadIdx.x + s] > 0)
            if (sh_dist[threadIdx.x] > sh_dist[threadIdx.x + s])
                sh_dist[threadIdx.x] = sh_dist[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        dist_r[blockIdx.x] = sh_dist[0];
}


__global__ void sum_reduce(double *dist_vec, double *dist_sum_r, const long long int n) {
    __shared__ double partial_sum[SHMEM_SIZE];
    long long int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
        partial_sum[threadIdx.x] = dist_vec[tid];
    else partial_sum[threadIdx.x] = 0;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s /= 2) {
        if (threadIdx.x < s)
            partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        dist_sum_r[blockIdx.x] = partial_sum[0];
}


void export_cities_to_csv(vector<city> &cities, string filename){
    ofstream file;
    file.open("../"+filename+".csv");
    if(file.is_open()){
        for(auto &e : cities){
            file << e.posX << "," << e.posY << "\n";
        }
        file.close();
    }
}


vector<city> generate_cities(long long int no_cities) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> distribution(1, MAX_X);
    vector<city> cities;
    for (long long int i = 0; i < no_cities; i++) {
        bool unique_city;
        city c{};
        do {
            unique_city = true;
            c.posX = (int)ceil(distribution(gen));
            c.posY = (int)ceil(distribution(gen));
            for (auto &e : cities) {
                if (e.posX == c.posX && e.posY == c.posY) unique_city = false;
            }
        } while (!unique_city);
        c.visited = false;
        cities.push_back(c);
    }
    return cities;
}


double distance(city cityA, city cityB) {
    return sqrt(pow(cityA.posX - cityB.posX, 2) + pow(cityA.posY - cityB.posY, 2));
}


double total_distance(vector<city> vec) {
    double total_dist = 0;
    for (long long int i = 0; i < vec.size() - 1; i++)
        total_dist += distance(vec[i], vec[i + 1]);

    return total_dist;
}


double nn_algorithm_cpu(vector<city> vec, long long int starting_point) {
    vector<city> result_vec;
    double max_dist = sqrt(pow(MAX_X, 2) + pow(MAX_Y, 2));
    auto current_index = starting_point;
    vec[current_index].visited = true;
    result_vec.push_back(vec[current_index]);
    while (result_vec.size() != vec.size()) {
        double min_dist = max_dist;
        long long int min_dist_index;
        for (long long int i = 0; i < vec.size(); i++) {
            if (i == current_index || vec[i].visited)
                continue;
            auto dist = distance(vec[current_index], vec[i]);
            if (dist < min_dist) {
                min_dist = dist;
                min_dist_index = i;
            }
        }
        current_index = min_dist_index;
        vec[current_index].visited = true;
        result_vec.push_back(vec[current_index]);
    }
    result_vec.push_back(result_vec[0]);
    export_cities_to_csv(result_vec, "sorted_cities_cpu");
    return total_distance(result_vec);
}

void set_visited_flag_to_false(vector<city> &cities){
    for(auto & c : cities)
        c.visited = false;
}


times compere_nn_algorithm(vector<city> cities, unsigned int starting_point){
    times nn_algorithm_times{};
    int n = cities.size();
    size_t cities_bytes = n * sizeof(city);
    size_t dist_bytes = n * sizeof(double);
    cudaEvent_t start_gpu;
    cudaEvent_t end_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&end_gpu);

    vector<city> sorted_cities;
    vector<double> h_dist(n);
    vector<double> h_dist_r(n);
    vector<double> h_min_dists;
    vector<double> h_total_dist(n);
    city *d_cities;
    double *d_dist;
    double *d_dist_r;
    double *d_min_dists;
    double *d_min_dist_sum;

    cudaMalloc(&d_min_dists, dist_bytes);
    cudaMalloc(&d_min_dist_sum, dist_bytes);
    cudaMalloc(&d_cities, cities_bytes);
    cudaMalloc(&d_dist, dist_bytes);
    cudaMalloc(&d_dist_r, dist_bytes);

    cudaMemcpy(d_cities, cities.data(), cities_bytes, cudaMemcpyHostToDevice);

    int threads = THREAD_SIZE;
    int blocks = (int) ceil((float) n / (float) threads);
    auto current_index = starting_point;
    sorted_cities.push_back(cities[current_index]);

    cudaEventRecord(start_gpu);
    while (sorted_cities.size() != cities.size()) {
        calculate_dist<<<blocks, threads>>>(d_cities, d_dist, current_index, n);
        find_min_reduction<<<blocks, threads>>>(d_dist, d_dist_r, n);
        find_min_reduction<<<1, threads>>>(d_dist_r, d_dist_r, n);

        cudaMemcpy(h_dist.data(), d_dist, dist_bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_dist_r.data(), d_dist_r, dist_bytes, cudaMemcpyDeviceToHost);
        h_min_dists.push_back(h_dist_r[0]);
        auto it = find(h_dist.begin(), h_dist.end(), h_dist_r[0]);
        current_index = it - h_dist.begin();
        sorted_cities.push_back(cities[current_index]);
    }

    sorted_cities.push_back(sorted_cities[0]);
    h_min_dists.push_back(distance(sorted_cities[n - 1], sorted_cities[0]));

    cudaMemcpy(d_min_dists, h_min_dists.data(), dist_bytes, cudaMemcpyHostToDevice);

    sum_reduce<<<blocks, threads>>>(d_min_dists, d_min_dist_sum, n);
    sum_reduce<<<1, threads>>>(d_min_dist_sum, d_min_dist_sum, n);

    cudaEventRecord(end_gpu);
    cudaEventSynchronize(end_gpu);

    float sorting_ms = 0;
    cudaEventElapsedTime(&sorting_ms, start_gpu, end_gpu);
    nn_algorithm_times.gpu_time = sorting_ms/1000;

    cudaMemcpy(h_total_dist.data(), d_min_dist_sum, dist_bytes, cudaMemcpyDeviceToHost);

    set_visited_flag_to_false(cities);

    auto start = chrono::high_resolution_clock::now();
    auto total_dist_cpu = nn_algorithm_cpu(cities, starting_point);
    auto end = chrono::high_resolution_clock::now();
    nn_algorithm_times.cpu_time = (double)chrono::duration_cast<chrono::nanoseconds>(end - start).count()/1000000000;
    nn_algorithm_times.time_diff = nn_algorithm_times.cpu_time/nn_algorithm_times.gpu_time;

    cout << "CPU total distance: " << total_dist_cpu << endl;
    cout << "GPU total distance: " << h_total_dist[0] << endl;

    export_cities_to_csv(sorted_cities, "sorted_cities_gpu");

    cudaFree(d_cities);
    cudaFree(d_dist);
    cudaFree(d_dist_r);
    cudaFree(d_min_dist_sum);
    cudaFree(d_min_dists);

    return nn_algorithm_times;
}

int main() {
    long long int n;
    cout << "n: " << endl;
    cin >> n;
    auto cities = generate_cities(n);
    unsigned int starting_point = 1;
    cout << "wygenerowano" << endl;

    auto nn_algorithm_times = compere_nn_algorithm(cities, starting_point);

    cout << "CPU runtime: " << nn_algorithm_times.cpu_time << "seconds"<< endl;
    cout << "GPU runtime: " << nn_algorithm_times.gpu_time << " seconds"<< endl;
    cout << "GPU did runtime test: " << nn_algorithm_times.time_diff << " faster" << endl;
    return 0;
}
