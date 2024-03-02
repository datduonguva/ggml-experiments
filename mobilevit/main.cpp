#include "ggml/ggml.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "common.h"
#include <iostream>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <map>

struct mobilevit_hparams {
    int num_channels = 3;
    int image_size = 256;
    int patch_size = 2;
    int hidden_sizes[3] = {144, 192, 240};
    int neck_hidden_sizes[7] = {16, 32, 64, 96, 128, 160, 640};
    int num_attention_heads = 4;
    float mlp_ratio = 2.0;
    float expand_ratio = 4.0;
    std::string hidden_act = "silu";
    int conv_kernel_size = 3;
    int output_stride = 32;
    float hidden_dropout_prob = 0.1;
    float attention_probs_dropout_prob = 0.0;
    float classifier_dropout_prob = 0.1;
    float initializer_range=0.02;
    float layer_norm_eps = 1e-5;
    bool qkv_bias = true;
};

struct mobilevit_conv_layer {
    struct ggml_tensor * kernel;    
//    struct ggml_tensor * bias;    // this conv layer never use bias
    struct ggml_tensor * gamma;
    struct ggml_tensor * beta;
    struct ggml_tensor * moving_mean;
    struct ggml_tensor * moving_variance;
};

struct inverted_residual_layer {
    int in_channels;
    int out_channels;
    int strides;
    int dilation = 1;
    bool use_residual;
    mobilevit_conv_layer expand_1x1;
    mobilevit_conv_layer conv_3x3;
    mobilevit_conv_layer reduce_1x1;
};

struct mobile_net_layer {
    int in_channels;
    int out_channels;
    int num_stages;
    int strides;
    std::vector<inverted_residual_layer> residual_layers;
};

struct mobilevit_transformer_layer{
    // attention
    ggml_tensor * attention_query_kernel;
    ggml_tensor * attention_query_bias;

    ggml_tensor * attention_key_kernel;
    ggml_tensor * attention_key_bias;

    ggml_tensor * attention_value_kernel;
    ggml_tensor * attention_value_bias;

    ggml_tensor * attention_output_kernel;
    ggml_tensor * attention_output_bias;

    // intermediate
    ggml_tensor * intermediate_kernel;
    ggml_tensor * intermediate_bias;

    // output
    ggml_tensor * output_kernel;
    ggml_tensor * output_bias;

    // layernorm_before
    ggml_tensor * lb_gamma;
    ggml_tensor * lb_beta;

    // layernorm_after
    ggml_tensor * la_gamma;
    ggml_tensor * la_beta;
};

struct mobilevit_transformer {
    int in_channels;
    int out_channels;
    int num_stages;
    int hidden_size;
    std::vector<mobilevit_transformer_layer> layers;
};

struct mobile_vit_layer {
    int in_channels;
    int out_channels;
    int num_stages;
    int strides;
    int hidden_size;
    int dilation;

    inverted_residual_layer downsampling_layer;
    mobilevit_conv_layer  conv_kxk;
    mobilevit_conv_layer conv_1x1;
    mobilevit_transformer transformer;
    
    ggml_tensor * layernorm_alpha;
    ggml_tensor * layernorm_beta;
    mobilevit_conv_layer conv_projection;
    mobilevit_conv_layer fusion;
};

struct mobilevit_encoder {
    mobile_net_layer layer_1;        
    mobile_net_layer layer_2;


    mobile_vit_layer layer_3;
    mobile_vit_layer layer_4;
    mobile_vit_layer layer_5;
};

struct mobilevit_model {
    mobilevit_hparams hparams;

    mobilevit_conv_layer conv_stem; 
    mobilevit_encoder encoder;  

    struct ggml_context * ctx_w;    // context for model's weights
    std::map<std::string, ggml_tensor *> tensors;
};

int total_weights = 0;
void read_weights(ggml_tensor * tensor, ggml_context * ctx_w, std::ifstream &fin){
    total_weights += 1;
    int name_length, n_dims;
    // read name_length
    fin.read(reinterpret_cast<char *>(&name_length), sizeof(name_length));

    // read name
    std::string name(name_length, 0);
    fin.read(&name[0], name_length);
    std::cout << "name: " << name << " ";

    // read n_dims
    fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
    std::cout << "n_dims: " << n_dims << ". ";
    
    int dims[4];
    std::cout << "Dim: (";
    for (int i = 0; i < n_dims; i++){
        fin.read(reinterpret_cast<char *>(&dims[i]), sizeof(int));
        std::cout <<  dims[i];
        if (i == n_dims - 1) std::cout << ")\n"; else std::cout << ", ";
    }
    // read the kernel
    if (n_dims == 4){ 
        tensor = ggml_new_tensor_4d(ctx_w, GGML_TYPE_F32, dims[0], dims[1], dims[2], dims[3]);
    }else if (n_dims == 3){
        tensor = ggml_new_tensor_3d(ctx_w, GGML_TYPE_F32, dims[0], dims[1], dims[2]);
    }else if (n_dims == 2){
        tensor = ggml_new_tensor_2d(ctx_w, GGML_TYPE_F32, dims[0], dims[1]);
    }else if (n_dims == 1){
        tensor = ggml_new_tensor_1d(ctx_w, GGML_TYPE_F32, dims[0]);
    }


    fin.read(
        reinterpret_cast<char *>(tensor->data),
        ggml_nbytes(tensor)
    );
}
 
void read_all_weights(mobilevit_model& model, std::ifstream &fin){
    // First, read all the weights
    while (true){
        total_weights += 1;
        int name_length, n_dims;
        // read name_length
        fin.read(reinterpret_cast<char *>(&name_length), sizeof(name_length));

        // read name
        std::string name(name_length, 0);
        fin.read(&name[0], name_length);
        std::cout << "name: ***" << name << "*** ";

        // read n_dims
        fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
        std::cout << "n_dims: " << n_dims << ". ";
        
        int dims[4];
        std::cout << "Dim: (";
        for (int i = 0; i < n_dims; i++){
            fin.read(reinterpret_cast<char *>(&dims[i]), sizeof(int));
            std::cout <<  dims[i];
            if (i == n_dims - 1) std::cout << ")\n"; else std::cout << ", ";
        }
        // read the kernel
        auto ctx_w = model.ctx_w;
        ggml_tensor * tensor;
        if (n_dims == 4){ 
            tensor = ggml_new_tensor_4d(ctx_w, GGML_TYPE_F32, dims[0], dims[1], dims[2], dims[3]);
        }else if (n_dims == 3){
            tensor = ggml_new_tensor_3d(ctx_w, GGML_TYPE_F32, dims[0], dims[1], dims[2]);
        }else if (n_dims == 2){
            tensor = ggml_new_tensor_2d(ctx_w, GGML_TYPE_F32, dims[0], dims[1]);
        }else if (n_dims == 1){
            tensor = ggml_new_tensor_1d(ctx_w, GGML_TYPE_F32, dims[0]);
        }


        fin.read(
            reinterpret_cast<char *>(tensor->data),
            ggml_nbytes(tensor)
        );
        
        model.tensors[name] = tensor;
        name  = "alskdjfasdf";

        if (fin.eof()) {std::cout << "done loading" << "\n"; break;}

    }
}
 
// overloading read mobileit_conv_layer
void read_weights(
    mobilevit_conv_layer & layer,
    ggml_context * ctx_w,
    std::ifstream &fin,
    bool read_gamma=true,
    bool read_beta=true
){
    read_weights(layer.kernel, ctx_w, fin); 
    if (read_gamma) read_weights(layer.gamma, ctx_w, fin);
    if (read_beta) read_weights(layer.beta, ctx_w, fin);
}

void assign_weights(
    mobilevit_conv_layer & layer,
    std::string path,
    std::map<std::string, ggml_tensor *> tensors,
    bool read_gamma=true
){

    layer.kernel = tensors.at(path + "/convolution/kernel:0");
    if (read_gamma){
        layer.gamma = tensors.at(path + "/normalization/gamma:0");
        layer.beta = tensors.at(path + "/normalization/beta:0");
        layer.moving_mean = tensors.at(path + "/normalization/moving_mean:0");
        layer.moving_variance = tensors.at(path + "/normalization/moving_variance:0");
    }    

    std::cout << "Done: " << path << "\n";
}


// overloading read transformer layers
void assign_weights(
    mobilevit_transformer_layer & layer,
    std::string path,
    std::map<std::string, ggml_tensor*> & tensors,
    bool read_gamma=true
){

    layer.attention_query_kernel = tensors.at(path + "/attention/attention/query/kernel:0");
    layer.attention_query_bias = tensors.at(path + "/attention/attention/query/bias:0");

    layer.attention_key_kernel = tensors.at(path + "/attention/attention/key/kernel:0");
    layer.attention_key_bias = tensors.at(path + "/attention/attention/key/bias:0");

    layer.attention_value_kernel = tensors.at(path + "/attention/attention/value/kernel:0");
    layer.attention_value_bias = tensors.at(path + "/attention/attention/value/bias:0");

    layer.attention_output_kernel = tensors.at(path + "/attention/output/dense/kernel:0");
    layer.attention_output_bias = tensors.at(path + "/attention/output/dense/bias:0");


    // intermediate
    layer.intermediate_kernel = tensors.at(path + "/intermediate/dense/kernel:0");
    layer.intermediate_bias = tensors.at(path + "/intermediate/dense/bias:0");


    // output
    layer.output_kernel = tensors.at(path + "/output/dense/kernel:0");
    layer.output_bias = tensors.at(path + "/output/dense/bias:0");


    // layernorm_before
    layer.lb_gamma = tensors.at(path + "/layernorm_before/gamma:0");
    layer.lb_beta = tensors.at(path + "/layernorm_before/beta:0");

    // layernorm_after
    layer.la_gamma = tensors.at(path + "/layernorm_after/gamma:0");
    layer.la_beta = tensors.at(path + "/layernorm_after/beta:0");

    std::cout << "Done: " << path << "\n";
}

// overloading the read_weights of the mobile_vit_layer
void assign_weights(
    mobile_vit_layer &layer,
    std::string path,
    std::map<std::string, ggml_tensor*> & tensors
){
    auto & downsampling_layer = layer.downsampling_layer;
    assign_weights(downsampling_layer.expand_1x1, path + "/downsampling_layer/expand_1x1", tensors);
    assign_weights(downsampling_layer.conv_3x3, path + "/downsampling_layer/conv_3x3", tensors);
    assign_weights(downsampling_layer.reduce_1x1, path + "/downsampling_layer/reduce_1x1", tensors);
   
    assign_weights(layer.conv_kxk, path + "/conv_kxk", tensors);
    // this conv_1x1 doesn't have normalization, so, there is no gamma and betta parameters
    assign_weights(layer.conv_1x1, path + "/conv_1x1", tensors, false);

    // read the transformer layers:
    layer.transformer.layers.resize(layer.num_stages);
    for (int i = 0; i < layer.num_stages; i++){
        auto & transformer_layer = layer.transformer.layers[0]; 
        assign_weights(transformer_layer, path + "/transformer/layer." + std::to_string(i), tensors);
    }
    // read the layernorm
    layer.layernorm_alpha = tensors.at(path + "/layernorm/gamma:0");
    layer.layernorm_beta = tensors.at(path + "/layernorm/beta:0");

    // read the projection
    assign_weights(layer.conv_projection, path + "/conv_projection", tensors);

    // read the fusion
    assign_weights(layer.fusion, path + "/fusion", tensors);

    std::cout << "Done: " << path << "\n";
}

void print_shape(ggml_tensor* tensor){
    if (tensor == NULL){
        std::cout << "Tensor is empty" << std::endl;
        return;
    }

    std::cout << "Dims: (";
    int n_dims = ggml_n_dims(tensor);
    for(int i = 0; i < n_dims; i++){
        if(i != n_dims-1){
            std::cout << tensor->ne[i] << ", ";   
        } else{
            std::cout << tensor->ne[i] << ")\n";   
        }
    }
}

void load_model_v2(mobilevit_model & model, std::string model_path){
    auto fin = std::ifstream(model_path, std::ios::binary);
    if (!fin){
        std::cout << "Error opening file" << std::endl;
    }

    // First, load all the weights into a map <name, tensor>
    read_all_weights(model, fin); 


    // next put the tensor in the right locations for each layers

    // read layer conv_stem
    {
        assign_weights(model.conv_stem, "tf_mobile_vi_t_model/mobilevit/conv_stem", model.tensors);
    }

     // read encoder
    {
        // read layer_1
        {
            int in_channels = model.hparams.neck_hidden_sizes[0];
            int out_channels = model.hparams.neck_hidden_sizes[1];
            int strides = 1;
            int num_stages = 1;
            // set the parasm
            model.encoder.layer_1.in_channels = in_channels;
            model.encoder.layer_1.out_channels = out_channels;
            model.encoder.layer_1.num_stages = num_stages;
            model.encoder.layer_1.strides = strides;
            model.encoder.layer_1.residual_layers.resize(num_stages);

            std::string path_base = "tf_mobile_vi_t_model/mobilevit/encoder/layer.0/layer.";
            for (int i = 0; i < num_stages; i++){
                auto & residual_layer = model.encoder.layer_1.residual_layers[i];
                std::string layer_base = path_base + std::to_string(i);
                residual_layer.in_channels = in_channels;
                residual_layer.out_channels = out_channels;
                residual_layer.strides = i == 0 ? strides : 1;
                
                assign_weights(residual_layer.expand_1x1, layer_base + "/expand_1x1", model.tensors);  
                assign_weights(residual_layer.conv_3x3, layer_base + "/conv_3x3", model.tensors); 
                assign_weights(residual_layer.reduce_1x1, layer_base + "/reduce_1x1", model.tensors);

                // after the first residual layer, in_channels equals out_channels
                in_channels = out_channels;

            }
        }        

        // assigning weights for layer 2
        {
            int in_channels = model.hparams.neck_hidden_sizes[1];
            int out_channels = model.hparams.neck_hidden_sizes[2];
            int strides = 2;
            int num_stages = 3;
            // set the parasm
            model.encoder.layer_2.in_channels = in_channels;
            model.encoder.layer_2.out_channels = out_channels;
            model.encoder.layer_2.num_stages = num_stages;
            model.encoder.layer_2.strides = strides;
            model.encoder.layer_2.residual_layers.resize(num_stages);

            std::string path_base = "tf_mobile_vi_t_model/mobilevit/encoder/layer.1/layer.";
            for (int i = 0; i < num_stages; i++){
                auto &residual_layer = model.encoder.layer_2.residual_layers[i];
                std::string layer_base = path_base + std::to_string(i);
                residual_layer.in_channels = in_channels;
                residual_layer.out_channels = out_channels;
                residual_layer.strides = i == 0 ? strides : 1;
                
                assign_weights(residual_layer.expand_1x1, layer_base + "/expand_1x1", model.tensors);
                assign_weights(residual_layer.conv_3x3, layer_base + "/conv_3x3", model.tensors);
                assign_weights(residual_layer.reduce_1x1, layer_base + "/reduce_1x1", model.tensors);
                // after the first residual layer, in_channels equals out_channels
                in_channels = out_channels;
            }
        }

        // read layer_3
        {
            int in_channels = model.hparams.neck_hidden_sizes[2];
            int out_channels = model.hparams.neck_hidden_sizes[3];
            int strides = 2;
            int num_stages = 2;
            int hidden_size = model.hparams.hidden_sizes[0];

            model.encoder.layer_3.in_channels = in_channels;
            model.encoder.layer_3.out_channels = out_channels;
            model.encoder.layer_3.num_stages = num_stages;
            model.encoder.layer_3.strides = strides;
            model.encoder.layer_3.hidden_size = hidden_size;
            model.encoder.layer_3.dilation = 1;

            assign_weights(
                model.encoder.layer_3,
                "tf_mobile_vi_t_model/mobilevit/encoder/layer.2",
                model.tensors
            );

        }

        // read layer 4
        {
            int in_channels = model.hparams.neck_hidden_sizes[3];
            int out_channels = model.hparams.neck_hidden_sizes[4];
            int strides = 2;
            int num_stages = 4;
            int hidden_size = model.hparams.hidden_sizes[1];

            model.encoder.layer_4.in_channels = in_channels;
            
            model.encoder.layer_4.out_channels = out_channels;
            model.encoder.layer_4.num_stages = num_stages;
            model.encoder.layer_4.strides = strides;
            model.encoder.layer_4.hidden_size = hidden_size;
            model.encoder.layer_4.dilation = 1;

            assign_weights(
                model.encoder.layer_4,
                "tf_mobile_vi_t_model/mobilevit/encoder/layer.3",
                model.tensors
            );
        }

        // read layer 5
        {
            int in_channels = model.hparams.neck_hidden_sizes[4];
            int out_channels = model.hparams.neck_hidden_sizes[5];
            int strides = 2 ;
            int num_stages = 3;
            int hidden_size = model.hparams.hidden_sizes[2];

            model.encoder.layer_4.in_channels = in_channels;
            
            model.encoder.layer_4.out_channels = out_channels;
            model.encoder.layer_4.num_stages = num_stages;
            model.encoder.layer_4.strides = strides;
            model.encoder.layer_4.hidden_size = hidden_size;
            model.encoder.layer_4.dilation = 1;

            assign_weights(
                model.encoder.layer_4,
                "tf_mobile_vi_t_model/mobilevit/encoder/layer.4",
                model.tensors
            );
        }


    }
}

struct sam_image_u8 {
    int nx, ny;
    std::vector<uint8_t> data;
};

bool sam_image_load_from_file(
    std::string &name,
    sam_image_u8 &img
){
    int nx, ny, nc;
    auto data = stbi_load(name.c_str(), &nx, &ny, &nc, 3);

    if (!data){
        std::cout << "Failed to load data\n";
        return false;
    }

    img.nx = nx;
    img.ny = ny;
    img.data.resize(nx * ny * 3);
    memcpy(img.data.data(), data, nx * ny * 3);
    stbi_image_free(data);
    return true;
}


int main(int argc, char ** argv) {

    // load image
    sam_image_u8 img0;
    std::string image_path = "/home/duongquocdat7411/tmp/sin(x).jpg";

    if (!sam_image_load_from_file(image_path, img0)){
        std::cout << "Failed to load image from file\n";
    }
    std::cout << "Size: "<< img0.nx << ", " << img0.ny << ", " << img0.data.size() << std::endl;
    ggml_time_init();
    mobilevit_model model;

    // create ggml_context for model's weight
    {
        struct ggml_init_params params = {128*1024*1024, NULL, false};

        model.ctx_w = ggml_init(params);
        if (!model.ctx_w) {
            std::cout << "Cannot create context for model's weights" << std::endl;
        }
    }

//    load_model_v2(model, "weight.ggml");

//    std::cout << "Total weights: " << total_weights << std::endl;
    return 0;
}
