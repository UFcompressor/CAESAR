#pragma once
#include <torch/script.h>   
#include <torch/autograd.h>
#include <iostream>       
#include <cmath>      
#include <torch/torch.h>
#include <optional>   
#include <functional>
#include <type_traits>
#include <vector>
#include <iterator>
#include <cstddef>
#include <stdexcept>

//Sorry to anyone who has to go through this code 
//I would normally never do this but python magic 
//I can truely test this though so I trust the magic
// BE careful you are messing with magic here ------------------------
template <typename T>
inline bool exists(const std::optional<T>& x) {
    return x.has_value();
}

template <typename T>
inline bool exists(const T* ptr) {
    return ptr != nullptr;
}

// I hate templates so much
inline bool exists(const torch::Tensor& t) {
    return t.defined();
}

// I guess disasbled this and try to understand the assembly idk how to debug this
// idk how this works I really hate python for making do this
//  seems to work and i trust it with the test
template <typename T, typename D>
inline T defaultVaule(const std::optional<T>& val, D&& d){
    if(exists(val)){
        return *val;
    }
    // d is a function or a lambda
    if constexpr (std::is_invocable_v<D>){
            return d();
     }
       else{
       // d is a normal value
            return d;
       }
}


// sorry for this hell bring me back to python
//  but it passes the test
template<typename Container>
class cycle {
private:
    const Container& dl;
    typename Container::const_iterator current;
    bool first_iteration;

public:
    cycle(const Container& container) : dl(container), first_iteration(true) {
        current = dl.begin();
    }
    
    typename Container::value_type operator()() {

        if (dl.empty()) {
            throw std::runtime_error("Cannot cycle through empty container");
        }
        

        if (current == dl.end()) {
            current = dl.begin();
        }
        
        auto value = *current;
        ++current;
        return value;
    }
}; 
       

template<typename Container>
cycle<Container> make_cycle(const Container& dl) {
    return cycle<Container>(dl);
} 
         

std::vector<int> numToGroups(int num, int divisor);



torch::Tensor extract(const torch::Tensor& a, const torch::Tensor& t, const std::vector<int64_t>& x_shape);







