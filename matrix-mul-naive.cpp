/*
Copyright (c) 2022, Akshat

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation 
and/or other materials provided with the distribution.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, 
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE 
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS 
OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, 
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include<CL/sycl.hpp>
#include "dpc_common.hpp"
#include<fstream>
#include<iostream>

using namespace std;
using namespace sycl;

class my_selector : public device_selector{
public:
    int operator () (const device &dev) const override {
        if(dev.get_info<info::device::name>().find("Intel") != std::string::npos)
            return 1;
        return -1;
    }
};

ofstream fout;

queue construct_queue(string dev){
        
    if(dev == "gpu"){
        gpu_selector selector;
        queue q(selector);
        fout.open("Outputs/out-gpu.txt", std::ios_base::app);
        return q;
    }
    if(dev == "cpu"){
        cpu_selector selector;
        queue q(selector);
        fout.open("Outputs/out-cpu.txt", std::ios_base::app);
        return q;
    }
    else{
        std::cout<<"Falling back to default selector\n";
        default_selector selector;
        queue q(selector);
        return q;
    }
}

void init_arrays(queue &q, size_t N, buffer<float, 2> &A_b, buffer<float, 2> &B_b, buffer<float, 2> &C_b){
    q.submit([&](handler &h){
        accessor A_acc(A_b, h, write_only);
        accessor B_acc(B_b, h, write_only);
        accessor C_acc(C_b, h, write_only);
        
        h.parallel_for(range{N, N}, [=](id<2> idx){
            int i = idx[0];
            int j = idx[1];
            A_acc[i][j] = i+1;
            B_acc[i][j] = j+1;
            C_acc[i][j] = 0;
        });
    });
}

void multiply(queue &q, size_t N, buffer<float, 2> &A_b, buffer<float, 2> &B_b, buffer<float, 2> &C_b){
    auto evt = q.submit([&](handler &h){
        accessor A_acc(A_b, h, read_only);
        accessor B_acc(B_b, h, read_only);
        accessor C_acc(C_b, h);
        
        h.parallel_for(range{N, N}, [=](id<2> idx){
            int i = idx[0];
            int j = idx[1];
            for(int k=0; k<N; k++){
                C_acc[i][j] += A_acc[i][k] * B_acc[k][j];
            }
        });
    });
    
    evt.wait();
}

int main(int argc, char* argv[]){
    size_t N = atoi(argv[1]);
    std::string dev = argv[2];    
    
    queue q = construct_queue(dev);

    device my_device = q.get_device();
    std::cout<<"Device Name: "<<my_device.get_info<info::device::name>()<<'\n';
    
    buffer<float, 2> A_b{range{N, N}};
    buffer<float, 2> B_b{range{N, N}};
    buffer<float, 2> C_b{range{N, N}};
    
    double elapsed_p = 0;
    dpc_common::TimeInterval timer_p;
    
    init_arrays(q, N, A_b, B_b, C_b);
    multiply(q, N, A_b, B_b, C_b);
    host_accessor C_host_acc(C_b, read_only);
    
    elapsed_p += timer_p.Elapsed();
    std::cout << "Time elapsed for N = "<<N<<" : " << elapsed_p << " sec\n";
    
    fout<<elapsed_p<<", ";
    fout.close();
    
    for(int i=0; i<10; i++){
        cout<<C_host_acc[0][i]<<" ";
    }
    cout<<"\n\n\n";
    
    return 0;
}


