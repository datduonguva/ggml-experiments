#include <ggml.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "utils.cpp"
using namespace std;


void get_image_label(int id, vector<int>&labels, vector<vector<int>>&images, int &label, vector<int>& image){
    label = labels.at(id);
    image = images.at(id);
} 

void get_mini_batch(
    vector<int>& labels,
    vector<vector<int>>&images,
    vector<int>&batch_id,
    struct ggml_tensor *input,
    struct ggml_tensor *target
){
    int batch_size = input->ne[1]; 
    int image_size = images[0].size();
    for (int i = 0; i < batch_size; i++){
        // set the target
        ggml_set_f32_1d(target, labels[batch_id[i]]*batch_size + i, 1.0);

        // set the images 
        for (int j = 0; j < image_size; j++){
            ggml_set_f32_1d(input, j*batch_size + i, images[batch_id[i]][j]/128.0 - 1); 
        }
    }
}

void show_image(ggml_tensor * images, int index){

    int batch = images->ne[1];
    for(int i = 0;i < 28; i++){
        for(int j = 0; j < 28; j++){
            int id = i*28 + j;
            float pixel = ggml_get_f32_1d(images, id*batch + index);
            if (pixel < 128){
                cout << "  ";
            }else {
                cout << "X ";
            }
        }
        cout << endl;
    }
}

struct ggml_tensor * square_error_loss(
    struct ggml_context * ctx,
    struct ggml_tensor * a,
    struct ggml_tensor * b) {
    return ggml_sum(ctx, ggml_sqr(ctx, ggml_sub(ctx, a, b)));
}
    
int main(){
    // define the memory
    struct ggml_init_params model_params = {
        .mem_size = 128*1024*1024,
        .mem_buffer = NULL,
        .no_alloc = false
    };
    
    // define the optimzier
    struct ggml_opt_params opt_params = ggml_opt_default_params(GGML_OPT_LBFGS);
    opt_params.print_forward_graph = false;
    opt_params.print_backward_graph = false;
    opt_params.lbfgs.n_iter = 16;

    
    cout << "n_threads: " << opt_params.n_threads << endl;
     
    // create the graph
    int batch_size = 128;
    struct ggml_context * model_ctx = ggml_init(model_params);
    // layer 1 weight and bias
    struct ggml_tensor * layer1_w = ggml_new_tensor_2d(model_ctx, GGML_TYPE_F32, 28 * 28, 512);
    struct ggml_tensor * layer1_b = ggml_new_tensor_2d(model_ctx, GGML_TYPE_F32, 512, 1);

    ggml_set_param(model_ctx, layer1_w);
    ggml_set_param(model_ctx, layer1_b);


    // laer 2 weight and bias
    struct ggml_tensor * layer2_w = ggml_new_tensor_2d(model_ctx, GGML_TYPE_F32, 512, 10);
    struct ggml_tensor * layer2_b = ggml_new_tensor_2d(model_ctx, GGML_TYPE_F32, 10, 1);

    ggml_set_param(model_ctx, layer2_w); 
    ggml_set_param(model_ctx, layer2_b);


    // Randomize the weight:
    randomize_tensor(layer1_w, 2, layer1_w->ne);
    randomize_tensor(layer1_b, 2, layer1_b->ne);
    randomize_tensor(layer2_w, 2, layer2_w->ne);
    randomize_tensor(layer2_b, 2, layer2_b->ne);


    // Read the training images and their labels;
    vector<int> labels;
    vector<vector<int>> images;
    read_csv("mnist_train.csv", labels, images); 
    cout << "size: " << labels.size() << ", " << images.size() << endl;



    vector<int> mask;
    for (int i = 0; i < labels.size(); i++){
        mask.push_back(i);
    }

    for (int epoch = 0; epoch < 10; epoch++){
        // Minibatch implementation
        shuffle(mask);

        for (int step=0; step < images.size(); step += batch_size){
	    cout << "Epoch: " <<  epoch << " ";
            cout << "step: " << step << " ";
            vector<int> batch_id (mask.begin() + step, mask.begin() + step + batch_size);
 
            // get a batch of training data
            
            struct ggml_init_params params = {
                .mem_size = 128*1024*1024,
                .mem_buffer = NULL,
                .no_alloc = false
            };

            struct ggml_context * ctx0 = ggml_init(params);
    
            // define the forward function [128, N]
            // The first layer
            struct ggml_tensor * input = ggml_new_tensor_2d(
                ctx0, GGML_TYPE_F32, 28 * 28,  batch_size
            );
            struct ggml_tensor * target = ggml_new_tensor_2d(
                ctx0, GGML_TYPE_F32, 10, batch_size
            );

            get_mini_batch(labels, images, batch_id, input, target); 

            // This is to show the training data
//            for (int i = 0; i < target->ne[0]; i++){
//                cout << "target at " << i << ": " << ggml_get_f32_1d(target, i) << endl;
//                show_image(input, i);
//            }

	    // this is basically the forward function
            struct ggml_tensor * x = ggml_mul_mat(ctx0, layer1_w, input);
            x = ggml_add(ctx0,
                x,
                ggml_repeat(ctx0, layer1_b, x)
            );
            x = ggml_relu(ctx0, x);

            // The second layer
            x = ggml_mul_mat(ctx0, layer2_w, x);

            struct ggml_tensor * logits = ggml_add(ctx0,
                x,
                ggml_repeat(ctx0, layer2_b, x)
            );

	    // build the graph, then get the logits 
            ggml_cgraph gf = {};
            ggml_build_forward_expand(&gf, logits);
	    ggml_graph_compute_with_ctx(ctx0, &gf, 1);  //last parameters is n_thread;
    
	    // calculate the loss
	    struct ggml_tensor * e = square_error_loss(ctx0, target, logits);

	    ggml_build_forward_expand(&gf, e);
	    ggml_graph_compute_with_ctx(ctx0, &gf, 1);
	    cout << " Loss before: " << ggml_get_f32_1d(e, 0) << " ";;
			    
	    // apply optimization
	    ggml_opt(ctx0, opt_params, e);

	    // check the loss after optimization
	    ggml_build_forward_expand(&gf, e);
	    ggml_graph_compute_with_ctx(ctx0, &gf, 1);
	    cout << "Loss After: " << ggml_get_f32_1d(e, 0) << endl;
	    
	    ggml_free(ctx0);
        }
    }
    
    return 0;
}
