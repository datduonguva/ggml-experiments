#include <ggml.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "utils.cpp"
using namespace std;


void get_image_label(int id, vector<int>*labels, vector<vector<int>>* images, int &label, vector<int>& image){
    label = labels->at(id);
    image = images->at(id);
} 

int main(){
    // define the memory
    struct ggml_init_params params ={
	.mem_size = 128*1024*1024,
	.mem_buffer = NULL,
	.no_alloc = false
    };
    
    // define the optimzier
    struct ggml_opt_params opt_params = ggml_opt_default_params(GGML_OPT_LBFGS);
    
    cout << "n_threads: " << opt_params.n_threads << endl;
     
    // create the graph
    int batch_size = 4;
    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_tensor * input = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 28 * 28,  batch_size);

    // layer 1 weight and bias
    struct ggml_tensor * layer1_w = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 28 * 28, 128);
    struct ggml_tensor * layer1_b = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 128, 1);

    ggml_set_param(ctx0, layer1_w);
    ggml_set_param(ctx0, layer1_b);


    // laer 2 weight and bias
    struct ggml_tensor * layer2_w = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 128, 10);
    struct ggml_tensor * layer2_b = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 10, 1);

    ggml_set_param(ctx0, layer2_w); 
    ggml_set_param(ctx0, layer2_b);

    // define the forward function [128, N]
    // The first layer
    struct ggml_tensor * x = ggml_mul_mat(ctx0,
	layer1_w,
	input
    );

    x = ggml_add(ctx0,
	x,
	ggml_repeat(ctx0, layer1_b, x)
    );
    x = ggml_relu(ctx0, x);

    // The second layer
    x = ggml_mul_mat(ctx0,
	layer2_w,
	x
    );

    x = ggml_add(ctx0,
	x,
	ggml_repeat(ctx0, layer2_b, x)
    );

    // Randomize the weight:
    randomize_tensor(layer1_w, 2, layer1_w->ne);
    randomize_tensor(layer1_b, 2, layer1_b->ne);
    randomize_tensor(layer2_w, 2, layer2_w->ne);
    randomize_tensor(layer2_b, 2, layer2_b->ne);


    // Read the training images and their labels;
    vector<int> labels;
    vector<vector<int>> images;
    read_csv("mnist_test.csv", labels, images); 
    cout << "size: " << labels.size() << ", " << images.size() << endl;



    vector<int> mask;
    for (int i = 0; i < labels.size(); i++){
	mask.push_back(i);
    }

    for (int i = 0; i < 100; i++){
	cout << mask[i] << ", ";
    }

    cout << endl;

    for (int epoch = 0; epoch < 1; epoch++){
	// Minibatch implementation
	shuffle(mask);
	for (int step=0; step < 100; step += batch_size){
	    cout << "step: " << step << endl;
	    vector<int> batch_id (mask.begin() + step, mask.begin() + step + batch_size);
	    // TODO: I am right here
	}
    }
    
    return 0;
}
