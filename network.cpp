#include "network.hpp"
#include "functions.hpp"

network::network(const vector<int> &dims){
    n_layers      = (int)dims.size();
    n_outnodes    = dims.back();
    identitymat   = arma::mat(n_outnodes, n_outnodes, arma::fill::eye);
    weights       = vector<arma::mat>(n_layers);
    biases        = vector<arma::vec>(n_layers);
    errors        = vector<arma::vec>(n_layers);
    activations   = vector<arma::vec>(n_layers);
    d_activations = vector<arma::vec>(n_layers);
    nodevalues    = vector<arma::vec>(n_layers);
    weightsdeltas = vector<arma::mat>(n_layers);
    biasesdeltas  = vector<arma::vec>(n_layers);
    
    for(int i = 1; i < n_layers; i++){
        double scale = sqrt((double)1/(double)dims.at(i-1));
        arma::arma_rng::set_seed_random();
        weights.at(i)       = arma::mat(dims.at(i), dims.at(i-1), arma::fill::randn) * scale;
        biases.at(i)        = arma::vec(dims.at(i),               arma::fill::randn);
        weightsdeltas.at(i) = arma::mat(dims.at(i), dims.at(i-1)                   );
        biasesdeltas.at(i)  = arma::vec(dims.at(i)                                 );
        activations.at(i)   = arma::vec(dims.at(i)                                 );
        d_activations.at(i) = arma::vec(dims.at(i)                                 );
        errors.at(i)        = arma::vec(dims.at(i)                                 );
    }
}

double network::compute_activation(const double &input){
    return 1/(1+exp(-input));
}

double network::compute_d_activation(const double &input){
    return (1/(1+exp(-input)))*(1-(1/(1+exp(-input))));
}

void network::feedforward(const arma::vec &image){
    activations.at(0) = image;
    for(int i = 1; i < n_layers; i++){
        nodevalues.at(i) = (weights.at(i) * activations.at(i-1)) + biases.at(i);
        for(int j = 0; j < activations.at(i).size(); j++){
            activations.at(i).at(j)   =   compute_activation(nodevalues.at(i).at(j));
            d_activations.at(i).at(j) = compute_d_activation(nodevalues.at(i).at(j));
        }
    }
}

void network::backpropagate(const int &label){
    errors.back() = (activations.back() - identitymat.col(label)) % d_activations.back();
    for(int i = n_layers - 2; i > 0; i--){
        errors.at(i) = (weights.at(i+1).t() * errors.at(i+1)) % d_activations.at(i);
    }
}

void network::stochastic_gradient_descent(const vector<arma::vec> &imageset, const vector<int> &labelset, const double &learnrate, const int &batchsize){
    double MSE = 0;
    double deltamultiplier  = learnrate/batchsize;
    unsigned long n_batches = imageset.size()/batchsize;
    for(int r = 0; r < n_batches; r++){
        for(int i = 0; i < batchsize; i++){
            feedforward(imageset.at(r*batchsize+i));
            backpropagate(labelset.at(r*batchsize+i));
            MSE += computecost();
            for(int j = 1; j < n_layers; j++){
                weightsdeltas.at(j) += errors.at(j) * activations.at(j-1).t();
                biasesdeltas.at(j)  += errors.at(j);
            }
        }
        for(int j = 1; j < n_layers; j++){
            weights.at(j) -= deltamultiplier * weightsdeltas.at(j);
            biases.at(j)  -= deltamultiplier * biasesdeltas.at(j);
            weightsdeltas.at(j).zeros();
            biasesdeltas.at(j).zeros();
        }
    }
    MSE /= imageset.size();
    cout << setprecision(8) << "train cost:\t" << MSE << endl;
    
}

double network::computecost(){
    double errornorm = arma::norm(errors.back());
    return pow(errornorm, 2)/2;
}

void network::test(const vector<arma::vec> &testimages, const vector<int> &testlabels){
    unsigned long n_tests = testimages.size();
    double MSE = 0;
    int count = 0;
    for(int j = 0; j < n_tests; j++){
        int digit = 0;
        double currentmax = 0;
        feedforward(testimages.at(j));
        MSE += computecost();
        for(int k = 0; k < 10; k++){
            if(currentmax < activations.back().at(k)){
                digit = k;
                currentmax = activations.back().at(k);
            }
        }
        if(digit == testlabels.at(j)) count++;
    }
    MSE /= n_tests;
    cout << setprecision(8) << "test cost:\t" << MSE << "\tscore: " << count << "/" << n_tests << endl;

}
