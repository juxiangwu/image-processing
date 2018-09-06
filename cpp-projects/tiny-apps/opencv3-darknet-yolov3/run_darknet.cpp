extern "C" {
#include <darknet.h>
}
static network *net;
#include <iostream>
void init_net(char *cfgfile,char *weightfile,int *inw,
              int *inh,int *outw,int *outh){
    net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    *inw = net->w;
    *inh = net->h;
    std::cout << "outw:"<<net->layers[net->n - 2].out_w<<std::endl;
    std::cout << "outh:"<<net->layers[net->n - 2].out_h<<std::endl;
    *outw = net->layers[net->n - 2].out_w;
    *outh = net->layers[net->n - 2].out_h;
}

float *run_net(float *indata)
{
    network_predict(net, indata);
    return net->output;
}
