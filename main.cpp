#include "functions.hpp"
#include "network.hpp"

using namespace std;

int main(void){
    
    //----------------------------------------------------------------------------------------------------------------------------------------//
    // LOAD DATA                                                FILEPATHS FOR:
    string IDXTRAINIMAGES {".../train-images-idx3-ubyte"};   // TRAINING IMAGES
    string IDXTRAINLABELS {".../train-labels-idx1-ubyte"};   // TRAINING LABELS
    string IDXTESTIMAGES  {".../t10k-images-idx3-ubyte"};    // TESTING IMAGES
    string IDXTESTLABELS  {".../t10k-labels-idx1-ubyte"};    // TESTING LABELS
    
    vector<arma::vec>                                   trainimages;
    vector<int>                                         trainlabels;
    vector<arma::vec>                                   testimages;
    vector<int>                                         testlabels;
    read_mnist_cv(IDXTRAINIMAGES.c_str(), IDXTRAINLABELS.c_str(), trainimages, trainlabels);
    read_mnist_cv(IDXTESTIMAGES.c_str(),  IDXTESTLABELS.c_str(),  testimages,  testlabels);
    
    
    //----------------------------------------------------------------------------------------------------------------------------------------//
    // INSTANTIATE NETWORK
    vector<int> NETWORK_DIMENSIONS      {28*28, 30, 10};    // SPECIFY NETWORK DIMENSIONS
    network                             network(NETWORK_DIMENSIONS);
    
    //----------------------------------------------------------------------------------------------------------------------------------------//
    // TRAIN AND MEASURE ACCURACY
    int EPOCHS = 100;    // SPECIFY NUMBER OF TRAINING EPOCHS
    for(int e = 0; e < EPOCHS; e++){
        cout << endl << "epoch: " << e+1 << endl;
        network.stochastic_gradient_descent(trainimages, trainlabels, 0.3, 10);
        shuffledata(trainimages, trainlabels);
        network.test(testimages, testlabels);
    }
    
    return 0;
}
