#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

struct city{
    int posX;
    int posY;
    bool visited;
};

city random_city(){
    city rand_city{};
    rand_city.posX = rand()%100;
    rand_city.posY = rand()%100;
    rand_city.visited = false;
    return rand_city;
}


vector<city> generate_cities(unsigned short no_cities){
    vector<city> cities(no_cities);
    generate(cities.begin(), cities.end(), random_city);
    return cities;
}



int main() {
    unsigned short n = 20;
    auto cities = generate_cities(n);

    return 0;
}
