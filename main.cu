#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

using namespace std;

#define MAX_X 10
#define MAX_Y 10
#define MAX_DIST MAX_X*MAX_X + MAX_Y*MAX_Y + 1
#define THREAD_SIZE 512
#define SHMEM_SIZE 512

struct city{
    float posX;
    float posY;
    bool visited;
};


__global__ void calculate_dist(city* cities, float* dist, unsigned short starting_point, unsigned short n){
    unsigned short tid = blockIdx.x*blockDim.x + threadIdx.x;
    if(tid < n) {
        if (tid == 0) cities[starting_point].visited = true;
        city starting_city = cities[starting_point];
        if (!cities[tid].visited && tid != starting_point) {
            dist[tid] = sqrt(
                    pow(cities[tid].posX - starting_city.posX, 2) + pow(starting_city.posY - cities[tid].posY, 2)
                    );
        } else dist[tid] = MAX_DIST;
    }
}

__global__ void find_min_reduction(float* dist, float* dist_r, unsigned short n){
    __shared__ float sh_dist[SHMEM_SIZE];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
        sh_dist[threadIdx.x] = dist[tid];
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s /= 2) {
        if (threadIdx.x < s && sh_dist[threadIdx.x+s] > 0)
            if(sh_dist[threadIdx.x] > sh_dist[threadIdx.x + s])
                sh_dist[threadIdx.x] = sh_dist[threadIdx.x +s];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        dist_r[blockIdx.x] = sh_dist[0];
}


city random_city(){
    city rand_city{};
    rand_city.posX = rand()%MAX_X;
    rand_city.posY = rand()%MAX_Y;
    rand_city.visited = false;
    return rand_city;
}


vector<city> generate_cities(unsigned short no_cities){
    vector<city> cities(no_cities);
    generate(cities.begin(), cities.end(), random_city);
    return cities;
}


float distance(city cityA, city cityB){
    return sqrt(pow(cityA.posX-cityB.posX, 2)+pow(cityA.posY-cityB.posY, 2));
}


float total_distance(vector<city> vec){
    float total_dist = 0;
    for(int i=0; i<vec.size()-1; i++)
        total_dist += distance(vec[i], vec[i+1]);

    return total_dist;
}


vector<city> nn_algorithm_gpu(vector<city> vec, unsigned short starting_point){
    int n = vec.size();
    size_t cities_bytes = n * sizeof(city);
    size_t dist_bytes = n * sizeof(float);
    vector<city> sorted_cities;
    vector<float> h_dist(n);
    vector<float> h_dist_r(n);
    city* d_cities;
    float* d_dist;
    float* d_dist_r;
    cudaMalloc(&d_cities, cities_bytes);
    cudaMalloc(&d_dist, dist_bytes);
    cudaMalloc(&d_dist_r, dist_bytes);


    cudaMemcpy(d_cities, vec.data(), cities_bytes, cudaMemcpyHostToDevice);

    int threads = THREAD_SIZE;
    int blocks = (int)ceil((float)n/(float)threads);
    int current_index = starting_point;
    sorted_cities.push_back(vec[current_index]);

    while(sorted_cities.size()!=vec.size()) {
        calculate_dist<<<blocks, threads>>>(d_cities, d_dist, current_index, n);
        find_min_reduction<<<blocks, threads>>>(d_dist, d_dist_r, n);
        find_min_reduction<<<1, threads>>>(d_dist_r, d_dist_r, blocks);

        cudaMemcpy(h_dist.data(), d_dist, dist_bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_dist_r.data(), d_dist_r, dist_bytes, cudaMemcpyDeviceToHost);
        auto it = find(h_dist.begin(), h_dist.end(), h_dist_r[0]);
        current_index = it - h_dist.begin();
        sorted_cities.push_back(vec[current_index]);
    }
    sorted_cities.push_back(sorted_cities[0]);
    return sorted_cities;
}


vector<city> nn_algorithm_cpu(vector<city> vec, unsigned short starting_point){
    vector<city> result_vec;
    float max_dist = sqrt(pow(MAX_X, 2)+ pow(MAX_Y, 2));
    auto current_index = starting_point;
    vec[current_index].visited = true;
    result_vec.push_back(vec[current_index]);
    while(result_vec.size() != vec.size()){
        float min_dist = max_dist;
        unsigned short min_dist_index;
        for(int i=0; i<vec.size(); i++){
            if(i==current_index || vec[i].visited)
                continue;
            auto dist = distance(vec[current_index], vec[i]);
            if(dist < min_dist){
                min_dist = dist;
                min_dist_index = i;
            }
        }
        current_index = min_dist_index;
        vec[current_index].visited = true;
        result_vec.push_back(vec[current_index]);
    }
    result_vec.push_back(result_vec[0]);
    return result_vec;
}


int main() {
    unsigned short n = 5;
    auto cities = generate_cities(n);
    auto sorted_cites = nn_algorithm_cpu(cities, 2);
    cout << "total distance " << total_distance(sorted_cites) << endl;
    auto v = nn_algorithm_gpu(cities, 2);
    cout << "total distance gpu: " << total_distance(v) << endl;

    return 0;
}
