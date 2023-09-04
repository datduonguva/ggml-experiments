#include <ggml.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "utils.cpp"
using namespace std;


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
        // one-hot encode the target
        ggml_set_f32_1d(target, labels[batch_id[i]]*batch_size + i, 1.0);

        // set the images 
        for (int j = 0; j < image_size; j++){
            ggml_set_f32_1d(input, j*batch_size + i, (float) images[batch_id[i]][j]/128.0f - 1.0f); 
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

struct ggml_tensor * cross_entropy_loss(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b) {
    const float eps = 1e-3f;
    return
        ggml_sum(ctx,
            ggml_neg(ctx,
                ggml_sum_rows(ctx,
                    ggml_mul(ctx,
                        ggml_soft_max(ctx, a),
                        ggml_log(ctx,
                            ggml_add1(ctx,
                                ggml_soft_max(ctx, b),
                                ggml_new_f32(ctx, eps)))))));
}    
struct mlp {
    struct ggml_context * ctx;
    struct ggml_tensor * layer1_w;
    struct ggml_tensor * layer1_b;
    struct ggml_tensor * layer2_w;
    struct ggml_tensor * layer2_b;
    struct ggml_tensor * layer3_w;
    struct ggml_tensor * layer3_b;

};

struct mlp create_mlp(){
    // define the memory
    struct ggml_init_params model_params = {
        .mem_size = 256*1024*1024,
        .mem_buffer = NULL,
        .no_alloc = false
    };
 
    struct ggml_context * model_ctx = ggml_init(model_params);

    struct mlp model;
    model.ctx = model_ctx;

    // layer 1 weight and bias

    model.layer1_w = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, 28 * 28, 1024);
    model.layer1_b = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, 1024, 1);

    ggml_set_param(model.ctx, model.layer1_w);
    ggml_set_param(model.ctx, model.layer1_b);


    // laer 2 weight and bias
    model.layer2_w = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, 1024, 64);
    model.layer2_b = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, 64, 1);

    ggml_set_param(model.ctx, model.layer2_w); 
    ggml_set_param(model.ctx, model.layer2_b);


    // layer 3 weight and bias
    model.layer3_w = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, 64 , 10);
    model.layer3_b = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, 10, 1);

    ggml_set_param(model.ctx, model.layer3_w); 
    ggml_set_param(model.ctx, model.layer3_b);


    // Randomize the weight:
    randomize_tensor(model.layer1_w, 2, model.layer1_w->ne);
    randomize_tensor(model.layer1_b, 2, model.layer1_b->ne);
    randomize_tensor(model.layer2_w, 2, model.layer2_w->ne);
    randomize_tensor(model.layer2_b, 2, model.layer2_b->ne);
    randomize_tensor(model.layer3_w, 2, model.layer3_w->ne);
    randomize_tensor(model.layer3_b, 2, model.layer3_b->ne);
    
    return model;
}


struct ggml_tensor * forward(struct mlp * model, struct ggml_context *ctx0, struct ggml_tensor *input){
    struct ggml_tensor * x = ggml_mul_mat(ctx0, model->layer1_w, input);
    x = ggml_add(ctx0,
	x,
	ggml_repeat(ctx0, model->layer1_b, x)
    );
    x = ggml_relu(ctx0, x);

    // The second layer
    x = ggml_mul_mat(ctx0, model->layer2_w, x);
    x = ggml_add(ctx0,
	x,
	ggml_repeat(ctx0, model->layer2_b, x)
    );
    x = ggml_relu(ctx0, x);

    // The second layer
    x = ggml_mul_mat(ctx0, model->layer3_w, x);

    auto * logits = ggml_add(ctx0,
	x,
	ggml_repeat(ctx0, model->layer3_b, x)
    );

    return logits;
}

// find max index along dimension 0
int argmax_2d(struct ggml_tensor * arr, int dim1){
    int batch_size = arr->ne[1];
    int n_class = arr->ne[0];
    int max_index = -1;
    float max_prob = -1e5;
    for (int j = 0; j < n_class; j++){
	float current_prob = ggml_get_f32_1d(arr, j*batch_size + dim1);
	if (current_prob > max_prob){
	    max_prob = current_prob;
	    max_index = j;
	}
    } 
    return max_index;
}
	

float get_batch_accuracy(struct ggml_tensor * logits, struct ggml_tensor * target){
    int batch_size = target->ne[1];
    int n_class = target->ne[0];
    int counter = 0;
    for (int i = 0; i < batch_size; i++){
	int y_true = argmax_2d(target, i);
	int y_pred = argmax_2d(logits, i);
	if (y_true == y_pred) counter +=1;
    }
    return (float) (1.0f*counter)/batch_size;
}

int main(){
   
    // Read the training images and their labels;
    vector<int> labels;
    vector<vector<int>> images;
    read_csv("mnist_train.csv", labels, images); 
    cout << "size: " << labels.size() << ", " << images.size() << endl;


    // create the model
    struct mlp model = create_mlp();

    vector<int> mask;
    for (int i = 0; i < labels.size(); i++){
        mask.push_back(i);
    }

    int batch_size = 64;
    for (int epoch = 0; epoch < 10; epoch++){
        // Minibatch implementation
        shuffle(mask);

        for (int step=0; step < images.size(); step += batch_size){
	    if (step == 128) step = 0;
	    cout << "Epoch: " <<  epoch << " ";
            cout << "step: " << step << " ";
            vector<int> batch_id (mask.begin() + step, mask.begin() + step + batch_size);
 
            // get a batch of training data
            
            struct ggml_init_params params = {
                .mem_size = 256*1024*1024,
                .mem_buffer = NULL,
                .no_alloc = false
            };

            struct ggml_context * ctx0 = ggml_init(params);

	    struct ggml_tensor * input = ggml_new_tensor_2d(
		ctx0, GGML_TYPE_F32, 28 * 28,  batch_size
	    );
	    struct ggml_tensor * target = ggml_new_tensor_2d(
		ctx0, GGML_TYPE_F32, 10, batch_size
	    );


            get_mini_batch(labels, images, batch_id, input, target); 

	    struct ggml_tensor * logits = forward(&model, ctx0, input);      

	    // build the graph, then get the logits 
            ggml_cgraph gf = {};
   
	    // calculate the loss
	    struct ggml_tensor * e = cross_entropy_loss(ctx0, target, logits);
		    
	    // define the optimzier
	    struct ggml_opt_params opt_params = ggml_opt_default_params(GGML_OPT_LBFGS);
	    opt_params.print_forward_graph = false;
	    opt_params.print_backward_graph = false;
	    opt_params.lbfgs.n_iter = 40;

	    // apply optimization
	    ggml_opt(ctx0, opt_params, e);

	    // check the loss after optimization
	    ggml_build_forward_expand(&gf, e);
	    ggml_graph_compute_with_ctx(ctx0, &gf, 1);
	    cout << "Loss After: " << ggml_get_f32_1d(e, 0) ;
	    
	    cout << " acc: " << get_batch_accuracy(logits, target) << endl;
	
	    ggml_free(ctx0);
        }
    }
    
    return 0;
}
