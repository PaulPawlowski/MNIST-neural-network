#pragma once
#include <vector>
#define ARMA_USE_BLAS
#include <armadillo>
#include <opencv.hpp>

using namespace std;

class network{
public:
    //DATA
    int               n_layers;
    int               n_outnodes;
    arma::mat         identitymat;
    vector<arma::mat> weights;
    vector<arma::mat> weightsdeltas;
    vector<arma::vec> biases;
    vector<arma::vec> biasesdeltas;
    vector<arma::vec> nodevalues;
    vector<arma::vec> activations;
    vector<arma::vec> d_activations;
    vector<arma::vec> errors;
    
    //CONSTRUCTOR
    network(const vector<int> &dims);
    
    //METHODS
    void feedforward(const arma::vec &image);
    void backpropagate(const int &label);
    void stochastic_gradient_descent(const vector<arma::vec> &imageset, const vector<int> &labelset, const double &learnrate, const int &batchsize);
    double computecost();
    void test(const vector<arma::vec> &testimages, const vector<int> &testlabels);
    double compute_activation(const double &input);
    double compute_d_activation(const double &input);
    

};

struct datapoint{
    arma::vec image;
    int label;
};
