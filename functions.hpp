#pragma once
#include <iostream>
#include <random>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <opencv2/opencv.hpp>
#define ARMA_USE_BLAS
#include <armadillo>

using namespace std;

uint32_t swap_endian(uint32_t val);

void read_mnist_cv(const char* image_filename, const char* label_filename, vector<arma::vec> &images, vector<int> &labels);

void shuffledata(vector<arma::vec> &images, vector<int> &labels);

