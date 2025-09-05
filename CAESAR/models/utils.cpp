#include "utils.h"



 std::vector<int>numToGroups(int num, int divisor){
    if(divisor <= 0){
        throw std::invalid_argument("Divisor must be positive +");
    }
    if(num < 0){
        throw std::invalid_argument("Num must be non-negative");
    }

    int groups = num / divisor;
    int rem = num % divisor;

    std::vector<int> arr(groups, divisor);
    if(rem > 0){
        arr.push_back(rem);
    }

    return arr;
}
