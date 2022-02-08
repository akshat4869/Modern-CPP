#include<CL/sycl.hpp>
#include "dpc_common.hpp"

using namespace sycl;

class my_selector : public device_selector{
public:
    int operator () (const device &dev) const override {
        if(dev.get_info<info::device::name>().find("Intel") != std::string::npos)
            return 1;
        return -1;
    }
};



constexpr int N = 1000;

int main(){    
    gpu_selector selector;
    queue q(selector);
    device my_device = q.get_device();
    std::cout<<"Device Name: "<<my_device.get_info<info::device::name>()<<'\n';
    
    buffer<int, 2> A_b{range{N, N}};
    buffer<int, 2> B_b{range{N, N}};
    buffer<int, 2> C_b{range{N, N}};
    
    q.submit([&](handler &h){
        accessor A_acc(A_b, h, write_only);
        accessor B_acc(B_b, h, write_only);
        accessor C_acc(C_b, h, write_only);
        
        h.parallel_for(range{N, N}, [=](id<2> idx){
            int i = idx[0];
            int j = idx[1];
            A_acc[i][j] = 1;
            B_acc[i][j] = 2;
            C_acc[i][j] = 0;
        });
    }).wait();
    
    double elapsed_p = 0;
    dpc_common::TimeInterval timer_p;
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
    elapsed_p += timer_p.Elapsed();
    std::cout << "Time parallel: " << elapsed_p << " sec\n";
    host_accessor C_host_acc(C_b, read_only);
    for(int i=0; i<10; i++){
        std::cout<<C_host_acc[0][i]<<' ';
    }
    std::cout<<'\n';

    return 0;
}


