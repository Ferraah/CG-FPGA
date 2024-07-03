#include "Timer.hpp"

#include <algorithm>


void Timer::start(){
    _current_start = std::chrono::high_resolution_clock::now();
}

void Timer::stop(){
    auto current_stop = std::chrono::high_resolution_clock::now();
    _durations.push_back(current_stop - _current_start);
}

void Timer::reset(){
    _durations.clear();
}

void Timer::print_last_formatted() const {
    double time = Timer::get_last() ;
        if(time < 1e-6){
            std::clog << time*1e9 << " ns";
        }else if(time < 1e-3){
            std::clog << time*1e6 << " us";
        }else if(time < 1){
            std::clog << time*1e3 << " ms";
        }else{
            std::clog << time << " s";
        }
}


std::string Timer::get_last_formatted() const {
    double time = Timer::get_last() ;
        if(time < 1e-6){
            return std::to_string(time*1e9) + " ns";
        }else if(time < 1e-3){
            return std::to_string(time*1e6) + " us";
        }else if(time < 1){
            return std::to_string(time*1e3) + " ms";
        }else{
            return std::to_string(time) + " s";
        }
}



void Timer::print(std::string title) const{
    if(title.empty()) 
        std::clog << "\n---- Timings ----" << std::endl;
    else
        std::clog << "\n----" << title << "----" << std::endl;

    for(size_t i = 0; i < _durations.size(); ++i){
        
        double time = _durations[i].count();
        if(time < 1e-6){
            std::clog << i+1 << ": " << time*1e9 << " ns" << std::endl;
        }else if(time < 1e-3){
            std::clog << i+1 << ": " << time*1e6 << " us" << std::endl;
        }else if(time < 1){
            std::clog << i+1 << ": " << time*1e3 << " ms" << std::endl;
        }else{
            std::clog << i+1 << ": " << time << " s" << std::endl;
        }
    }
    std::clog << "-----------------\n" << std::endl;
}

double Timer::get_last() const{
    if(_durations.size() > 0 )
        return _durations.back().count();
    else
        return -1.0; 
}

double _count(Timer::duration dur){
    return dur.count();
}

double Timer::get_min() const{
    std::vector<double> counts(_durations.size());

    std::transform(_durations.begin(), _durations.end(), counts.begin(), _count);
    
    auto min = std::min_element(counts.begin(), counts.end());

    // @Todo: improve with exceptions
    if(min!=counts.end()){
        return *min;
    }else{
        return 0;
    }

}
