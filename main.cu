#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

using namespace std;

#define MAX_X 10000
#define MAX_Y 10000

struct city{
    float posX;
    float posY;
    bool visited;
};

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


vector<city> nn_algorithm(vector<city> vec, unsigned short starting_point){
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

    return result_vec;
}


int main() {
    unsigned short n = 10000;
    auto cities = generate_cities(n);

    return 0;
}
