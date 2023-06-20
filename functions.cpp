#include "functions.hpp"
#include "network.hpp"

// SOURCE REFERENCE: https://stackoverflow.com/questions/8286668/how-to-read-mnist-data-in-c
uint32_t swap_endian(uint32_t val) {
    val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
    return (val << 16) | (val >> 16);
}

// SOURCE REFERENCE: https://stackoverflow.com/questions/8286668/how-to-read-mnist-data-in-c
void read_mnist_cv(const char* image_filename, const char* label_filename, vector<arma::vec> &images, vector<int> &labels){
    // Open files
    std::ifstream image_file(image_filename, std::ios::in | std::ios::binary);
    std::ifstream label_file(label_filename, std::ios::in | std::ios::binary);

    // Read the magic and the meta data
    uint32_t magic;
    uint32_t num_items;
    uint32_t num_labels;
    uint32_t rows;
    uint32_t cols;

    image_file.read(reinterpret_cast<char*>(&magic), 4);
    magic = swap_endian(magic);
    if(magic != 2051){
        cout<<"Incorrect image file magic: "<<magic<<endl;
        return;
    }

    label_file.read(reinterpret_cast<char*>(&magic), 4);
    magic = swap_endian(magic);
    if(magic != 2049){
        cout<<"Incorrect image file magic: "<<magic<<endl;
        return;
    }

    image_file.read(reinterpret_cast<char*>(&num_items), 4);
    num_items = swap_endian(num_items);
    label_file.read(reinterpret_cast<char*>(&num_labels), 4);
    num_labels = swap_endian(num_labels);
    if(num_items != num_labels){
        cout<<"image file nums should equal to label num"<<endl;
        return;
    }

    image_file.read(reinterpret_cast<char*>(&rows), 4);
    rows = swap_endian(rows);
    image_file.read(reinterpret_cast<char*>(&cols), 4);
    cols = swap_endian(cols);

    char label;
    char* pixels = new char[rows * cols];
    
    for (int item_id = 0; item_id < num_items; ++item_id) {
        // read image pixels
        image_file.read(pixels, rows * cols);
        // read and pushback label
        label_file.read(&label, 1);
        labels.push_back((int)(label));
        // convert to cv::Mat
        cv::Mat image_tmp(rows, cols, CV_8UC1, pixels);
        // convert to arma::vec
        arma::vec imagevector(784);
        for(int row = 0; row < 28; row++){
            for(int col = 0; col < 28; col++){
                imagevector.at(row*28+col) = (((double)image_tmp.at<uchar>(row,col))/255);
            }
        }
        images.emplace_back(imagevector);
    }
    delete[] pixels;
}

void shuffledata(vector<arma::vec> &images, vector<int> &labels){
    vector<datapoint> dataset(images.size());
    for(int i = 0; i < images.size(); i++){
        dataset.at(i).image = images.at(i);
        dataset.at(i).label = labels.at(i);
    }
    long long seed = std::chrono::system_clock::now().time_since_epoch().count();
    shuffle(dataset.begin(), dataset.end(), default_random_engine((unsigned)seed));
    for(int i = 0; i < images.size(); i++){
        images.at(i) = dataset.at(i).image;
        labels.at(i) = dataset.at(i).label;
    }
}
