#include <ggml.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

using namespace std;

void read_csv(string file_name, vector<int> &labels, vector<vector<int>> &pixels){
    fstream input_file;
    input_file.open(file_name, ios::in);


    if (input_file.is_open()){
	cout << "The file is open" << endl;
        string sa;
    
	while(getline(input_file, sa)){
	    
	    int start_index = 0;
		
	    int label;
	    vector<int> pixel; 
	    vector<int> sep_position {0};

	    // get positions of the commas
	    for(int i=0; i<sa.length(); i++)
		if (sa[i] == ',') sep_position.push_back(i);
	    sep_position.push_back(sa.length());

	    //
	    label = stoi(sa.substr(sep_position[0], sep_position[1] - sep_position[0]));

	    for (int i = 1; i < sep_position.size() - 1; i++){
		// the number start as comma's position + 1
		pixel.push_back(stoi(sa.substr(sep_position[i] + 1, sep_position[i+1] - sep_position[i])));
	    }
		// the number start as comma's position + 1

	    labels.push_back(label);
	    pixels.push_back(pixel);
	}
    }
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

    cout << x->ne[0] <<", " << x->ne[1] << endl;
    x = ggml_add(ctx0,
	x,
	ggml_repeat(ctx0, layer2_b, x)
    );
    x = ggml

    cout << x->ne[0] <<", " << x->ne[1] << endl;

    


    // Load the training data
    if (0){
	vector<int> labels;
	vector<vector<int>> pixels;
	read_csv("mnist_test.csv", labels, pixels); 

	// build the model of 2 DNN of 3 DNN layers

	cout << "Label size: " << labels.size() << endl;
	cout << "Pixel size: " << pixels.size() << endl;
    }
    return 0;
}
