// DeepBasic.cpp : Defines the entry point for the console application.
//

#include <dlib/dnn.h>
#include <iostream>
#include <dlib/data_io.h>

int main()
{
    // load the data
    std::vector<dlib::matrix<unsigned char>> train_images;
    std::vector<unsigned long> train_labels;
    std::vector<dlib::matrix<unsigned char>> test_images;
    std::vector<unsigned long> test_labels;
    dlib::load_mnist_dataset("../data/mnistImages", train_images, train_labels,
        test_images, test_labels);

    // define leNet
    using net_type = dlib::loss_multiclass_log <
        dlib::fc < 10,   // fully-connected 10 output
        dlib::relu < dlib::fc < 84,  // fully-connected 84 output -> relu
        dlib::relu < dlib::fc < 120,  // fully-connected 120 output->relu
        dlib::max_pool<2, 2, 2, 2, dlib::relu<dlib::con<16, 5, 5, 1, 1, // convolute 5x5x16 ->relu-> maxpool
        dlib::max_pool<2, 2, 2, 2, dlib::relu<dlib::con<6, 5, 5, 1, 1, // convolute 5x5x6 -> relu->maxpool
        dlib::input<dlib::matrix<unsigned char>>  // input 28x28 image
        >>>>>>>>>>>>;
    net_type net;

    // train the net
    dlib::dnn_trainer<net_type> trainer(net);
    trainer.set_learning_rate(0.01);
    trainer.set_min_learning_rate(0.00001);
    trainer.set_mini_batch_size(128);
    trainer.set_max_num_epochs(5);
    trainer.be_verbose();

    // save results every 20 second
    trainer.set_synchronization_file("./DeepBasic.dir/mnist_sync", std::chrono::seconds(20));

    // start training
    trainer.train(train_images, train_labels);

    // save the trained model
    net.clean();
    dlib::serialize("./DeepBasic.dir/mnist_network.dat") << net;

    // now run train images through net
    std::vector<unsigned long> predicted_labels = net(train_images);
    int num_right = 0;
    int num_wrong = 0;
    for (size_t i =0; i< train_images.size(); ++i)
    {
        if (predicted_labels[i] == train_labels[i])
            ++num_right;
        else
            ++num_wrong;
    }
    std::cout << "training number right= " << num_right << std::endl;
    std::cout << "training number wrong= " << num_wrong << std::endl;
    std::cout << "training accuracy= " << num_right / double(num_right + num_wrong) << std::endl;

    // now run on test images
    predicted_labels.clear();
    predicted_labels = net(test_images);
    num_right = 0; num_wrong = 0;
    for (size_t i=0; i< predicted_labels.size(); ++i)
    {
        if (predicted_labels[i] == test_labels[i])
        {
            ++num_right;
        }
        else
        {
            ++num_wrong;
        }
    }
    std::cout << "test number right= " << num_right << "\n";
    std::cout << "test number wrong= " << num_wrong << "\n";
    std::cout << "test accuracy= " << num_right / double(num_right + num_wrong) << "\n";



    std::system("pause");

    return 0;
}
