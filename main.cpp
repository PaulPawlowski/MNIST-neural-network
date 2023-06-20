#include "functions.hpp"
#include "network.hpp"

using namespace std;

int main(void){
    
    //----------------------------------------------------------------------------------------------------------------------------------------//
    // LOAD AND ORGANIZE DATA
    string IDXIMAGES = "/Users/paulpawlowski/Documents/C++/nn2/data/train-images-idx3-ubyte";   // SPECIFY IMAGES IDX-FILE PATH
    string IDXLABELS = "/Users/paulpawlowski/Documents/C++/nn2/data/train-labels-idx1-ubyte";   // SPECIFY LABELS IDX-FILE PATH
    vector<arma::vec>                                   images;
    vector<int>                                         labels;
    vector<arma::vec>                                   trainimages;
    vector<int>                                         trainlabels;
    vector<arma::vec>                                   testimages;
    vector<int>                                         testlabels;
    
    read_mnist_cv(IDXIMAGES.c_str(), IDXLABELS.c_str(), images, labels);
    
    for(int i = 0; i < 50000; i++){
        trainimages.push_back(images.at(i));
        trainlabels.push_back(labels.at(i));
    }
    for(int i = 50000; i < 60000; i++){
        testimages.push_back(images.at(i));
        testlabels.push_back(labels.at(i));
    }
    images.clear();
    labels.clear();
    
    //----------------------------------------------------------------------------------------------------------------------------------------//
    // INSTANTIATE NETWORK
    vector<int> NETWORK_DIMENSIONS      {28*28, 30, 10};                                        // SPECIFY NETWORK DIMENSIONS
    network                             network(NETWORK_DIMENSIONS);
    
    //----------------------------------------------------------------------------------------------------------------------------------------//
    // TRAIN AND MEASURE ACCURACY
    int EPOCHS = 100;                                                                           // SPECIFY NUMBER OF TRAINING EPOCHS
    for(int e = 0; e < EPOCHS; e++){
        cout << endl << "epoch: " << e+1 << endl;
        network.stochastic_gradient_descent(trainimages, trainlabels, 0.3, 10);
        shuffledata(trainimages, trainlabels);
        network.test(testimages, testlabels);
    }
    
    return 0;
}


