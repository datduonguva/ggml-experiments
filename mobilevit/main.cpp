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


void print_shape(std::string name, ggml_tensor* tensor);

struct sam_image_f32 {                                                                              
    int nx;                                                                                         
    int ny;                                                                                         
                                                                                                    
    std::vector<float> data;                                                                        
}; 

struct sam_image_u8 {
    int nx, ny;
    std::vector<uint8_t> data;
};


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
    float layer_norm_eps = 1.0e-5;
    bool qkv_bias = true;
};


struct mobilevit_conv_layer {
    struct ggml_tensor * kernel;    
//    struct ggml_tensor * bias;    // this conv layer never use bias
    struct ggml_tensor * gamma;
    struct ggml_tensor * beta;
    struct ggml_tensor * moving_mean;
    struct ggml_tensor * moving_variance;

    // forward
    struct ggml_tensor * forward(
        ggml_context * ctx,
        ggml_tensor * input,
        int s = 1,
        bool use_normalization=true,
        bool use_activation=true,
        bool depthwise=false
    );
 
};

struct inverted_residual_layer {
    int in_channels;
    int out_channels;
    int strides = 1;
    int dilation = 1;
    bool use_residual;
    mobilevit_conv_layer expand_1x1;
    mobilevit_conv_layer conv_3x3;
    mobilevit_conv_layer reduce_1x1;
    
    // forward
    ggml_tensor * forward(ggml_context * ctx, ggml_tensor * inp);
};

struct mobile_net_layer {
    int in_channels;
    int out_channels;
    int num_stages;
    int strides;
    std::vector<inverted_residual_layer> residual_layers;

    // forward
    ggml_tensor * forward(
        ggml_context * ctx, ggml_tensor * inp
    ){
        for (int i = 0; i < num_stages; i++){
            inp = residual_layers[i].forward(ctx, inp);
        }
        return inp;  // placeholder
    }
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


    ggml_tensor * forward(ggml_context * ctx, ggml_tensor * inp, float layer_norm_esp, int num_head);
};

// a transformers has many transformer layers
struct mobilevit_transformer {
    int num_stages;
    int hidden_size;
    int num_head;
    std::vector<mobilevit_transformer_layer> layers;

    ggml_tensor * forward(ggml_context * ctx, ggml_tensor * inp, float layer_norm_esp);
};

struct mobile_vit_layer {
    int in_channels;
    int out_channels;
    int num_stages;
    int strides = 2;
    int patch_size = 2;
    int hidden_size;
    float layer_norm_esp;
    int dilation;

    inverted_residual_layer downsampling_layer;
    mobilevit_conv_layer conv_kxk;
    mobilevit_conv_layer conv_1x1;
    mobilevit_transformer transformer;
    
    ggml_tensor * layernorm_alpha;
    ggml_tensor * layernorm_beta;
    mobilevit_conv_layer conv_projection;
    mobilevit_conv_layer fusion;

    ggml_tensor * unfolding(ggml_context * ctx, ggml_tensor * features, int patch_size);
    ggml_tensor * folding(ggml_context * ctx, ggml_tensor * features, int patch_size);

    // forward
    ggml_tensor * forward( ggml_context * ctx, ggml_tensor * inp);
};

struct mobilevit_encoder {
    mobile_net_layer layer_1;        
    mobile_net_layer layer_2;

    mobile_vit_layer layer_3;
    mobile_vit_layer layer_4;
    mobile_vit_layer layer_5;

    ggml_tensor * forward(ggml_context * ctx, ggml_tensor * embedding){
        std::cout << "----------->through layer_1\n";
        ggml_tensor * result = layer_1.forward(ctx, embedding);
        std::cout << "----------->through layer_2\n";
        result = layer_2.forward(ctx, result);
        std::cout << "----------->through layer_3\n";
        result = layer_3.forward(ctx, result);
        std::cout << "----------->through layer_4\n";
        result = layer_4.forward(ctx, result);
        std::cout << "----------->through layer_5\n";
        result = layer_5.forward(ctx, result);
        return result;
    }
};

struct mobilevit_model {
    mobilevit_hparams hparams;

    mobilevit_conv_layer conv_stem; 
    mobilevit_encoder encoder;  
    mobilevit_conv_layer conv_1x1_exp;

    struct ggml_context * ctx_w;    // context for model's weights
    std::map<std::string, ggml_tensor *> tensors;

    ggml_tensor * extract_features(sam_image_f32 & img);
};

int total_weights = 0;
void read_all_weights(mobilevit_model& model, std::ifstream &fin);

void assign_weights(
    mobilevit_conv_layer & layer,
    std::string path,
    std::map<std::string, ggml_tensor *> tensors,
    bool use_normalization=true
){

    layer.kernel = tensors.at(path + "/convolution/kernel:0");
    if (use_normalization){
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
    bool use_normalization=true
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
        auto & transformer_layer = layer.transformer.layers[i]; 
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
            model.encoder.layer_3.patch_size = model.hparams.patch_size;
            model.encoder.layer_3.layer_norm_esp = model.hparams.layer_norm_eps;

            // inverted layer need to have their in_channels and out_channels set
            model.encoder.layer_3.downsampling_layer.in_channels = in_channels;
            model.encoder.layer_3.downsampling_layer.out_channels = out_channels;
            model.encoder.layer_3.downsampling_layer.strides = strides;
            
            // settings parameters for transformer
            model.encoder.layer_3.transformer.num_stages = num_stages;
            model.encoder.layer_3.transformer.hidden_size = hidden_size;
            model.encoder.layer_3.transformer.num_head = model.hparams.num_attention_heads;
            model.encoder.layer_3.transformer.layers.resize(num_stages); 


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
            model.encoder.layer_4.patch_size = model.hparams.patch_size;
            model.encoder.layer_4.layer_norm_esp = model.hparams.layer_norm_eps;

            // inverted layer need to have their in_channels and out_channels set
            model.encoder.layer_4.downsampling_layer.in_channels = in_channels;
            model.encoder.layer_4.downsampling_layer.out_channels = out_channels;
            model.encoder.layer_4.downsampling_layer.strides = strides;
     
            // settings parameters for transformer
            model.encoder.layer_4.transformer.num_stages = num_stages;
            model.encoder.layer_4.transformer.hidden_size = hidden_size;
            model.encoder.layer_4.transformer.num_head = model.hparams.num_attention_heads;
            model.encoder.layer_4.transformer.layers.resize(num_stages); 


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

            model.encoder.layer_5.in_channels = in_channels;
            
            model.encoder.layer_5.out_channels = out_channels;
            model.encoder.layer_5.num_stages = num_stages;
            model.encoder.layer_5.strides = strides;
            model.encoder.layer_5.hidden_size = hidden_size;
            model.encoder.layer_5.dilation = 1;
            model.encoder.layer_5.patch_size = model.hparams.patch_size;
            model.encoder.layer_5.layer_norm_esp = model.hparams.layer_norm_eps;

            // inverted layer need to have their in_channels and out_channels set
            model.encoder.layer_5.downsampling_layer.in_channels = in_channels;
            model.encoder.layer_5.downsampling_layer.out_channels = out_channels;
            model.encoder.layer_5.downsampling_layer.strides = strides;
 
            // settings parameters for transformer
            model.encoder.layer_5.transformer.num_stages = num_stages;
            model.encoder.layer_5.transformer.hidden_size = hidden_size;
            model.encoder.layer_5.transformer.num_head = model.hparams.num_attention_heads;
            model.encoder.layer_5.transformer.layers.resize(num_stages); 



            assign_weights(
                model.encoder.layer_5,
                "tf_mobile_vi_t_model/mobilevit/encoder/layer.4",
                model.tensors
            );
        }


    }
}

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


bool sam_image_preprocess(const sam_image_u8 & img, sam_image_f32 & res) {
    const int nx = img.nx;
    const int ny = img.ny;

    const int nx2 = 256;
    const int ny2 = 256;

    res.nx = nx2;
    res.ny = ny2;
    res.data.resize(3*nx2*ny2);

    const float scale = std::max(nx, ny)*1.0f /(float) nx2;

    fprintf(stderr, "%s: scale = %f\n", __func__, scale);

    const int nx3 = int(nx/scale + 0.5f);
    const int ny3 = int(ny/scale + 0.5f);

//    const float m3[3] = { 123.675f, 116.280f, 103.530f };
//    const float s3[3] = {  58.395f,  57.120f,  57.375f };
    const float m3[3] = {0.0f, 0.0f, 0.0f};
    const float s3[3] = {255.0f, 255.0f, 255.0f};

    for (int y = 0; y < ny3; y++) {                                                                 
        for (int x = 0; x < nx3; x++) {                                                             
            for (int c = 0; c < 3; c++) {                                                           
                // linear interpolation                                                             
                const float sx = (x + 0.5f)*scale - 0.5f;                                           
                const float sy = (y + 0.5f)*scale - 0.5f;                                           
                                                                                                    
                const int x0 = std::max(0, (int) std::floor(sx));                                   
                const int y0 = std::max(0, (int) std::floor(sy));                                   
                                                                                                    
                const int x1 = std::min(x0 + 1, nx - 1);                                            
                const int y1 = std::min(y0 + 1, ny - 1);                                            
                                                                                                    
                const float dx = sx - x0;                                                           
                const float dy = sy - y0;                                                           
                                                                                                    
                const int j00 = 3*(y0*nx + x0) + c;                                                 
                const int j01 = 3*(y0*nx + x1) + c;                                                 
                const int j10 = 3*(y1*nx + x0) + c;                                                 
                const int j11 = 3*(y1*nx + x1) + c;                                                 
                                                                                                    
                const float v00 = img.data[j00];                                                    
                const float v01 = img.data[j01];                                                    
                const float v10 = img.data[j10];                                                    
                const float v11 = img.data[j11];                                                    
                                                                                                    
                const float v0 = v00*(1.0f - dx) + v01*dx;                                          
                const float v1 = v10*(1.0f - dx) + v11*dx;                                          
                                                                                                    
                const float v = v0*(1.0f - dy) + v1*dy;                                             
                                                                                                    
                const uint8_t v2 = std::min(std::max(std::round(v), 0.0f), 255.0f);                 
                                                                                                    
                const int i = 3*(y*nx3 + x) + c;                                                    
                res.data[i] = (float(v2) - m3[c]) / s3[c];                                          
            }                                                                                       
        }                                                                                           
    }                                                                                               
                                                                                                    
    return true;                                                                                    
}


ggml_tensor * mobilevit_model::extract_features(sam_image_f32 & img){
    struct ggml_init_params params = { 1024* 1024 * 1024, NULL, false};

    ggml_context * ctx0 = ggml_init(params);
    ggml_cgraph * gf = ggml_new_graph(ctx0);

    // input to the model is the actual image

    ggml_tensor * inp = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 256, 256, 3, 1);

    ggml_set_name(inp, "inp");
    ggml_set_input(inp);

    // signature: ctx, inp, s, use_normalization, use_activation, depthwise
    ggml_tensor * output = conv_stem.forward(ctx0, inp, 2, true, true,false ); // [128, 128, 16]

    // call the encoder
    output = encoder.forward(ctx0, output);

    print_shape("final output: ", output);

    float * data = (float *) ggml_get_data(inp);
    for(int k = 0; k < 3;k ++){
        for(int y =0; y < 256; y++){
            for(int x = 0; x < 256; x++){
                data[k * 256 * 256 + y * 256 + x] = img.data[y * 3*  256 + x * 3 + k]; 
            }
        }
    }

    std::cout << "Before building grapth\n";
    ggml_build_forward_expand(gf, output);

    std::cout << "after building grapth\n";
    for (int i = 0; i < 10; i++){
        auto t0 = ggml_time_us();
        ggml_graph_compute_with_ctx(ctx0, gf, 4);
        std::cout << "end compute: " << (ggml_time_us() - t0)/1000.f << " (ms)\n";
    }
    ggml_free(ctx0);

    return output;
} 


int main(int argc, char ** argv) {

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

    // load model weights:
    load_model_v2(model, "weight.ggml");
    std::cout << "Total weights: " << total_weights << std::endl;

    // load image
    sam_image_u8 img0;
    sam_image_f32 img1;
    std::string image_path = "/home/datduong/Desktop/IMG_5034.JPG";

    if (!sam_image_load_from_file(image_path, img0)){
        std::cout << "Failed to load image from file\n";
    }
    std::cout << "Size: "<< img0.nx << ", " << img0.ny << ", " << img0.data.size() << std::endl;

    const int64_t t0 = ggml_time_us();
    sam_image_preprocess(img0, img1);

    const int64_t t_load_image = ggml_time_us();
    // calculate the graph
    ggml_tensor * gf = model.extract_features(img1);


    const int64_t t_encode = ggml_time_us();
    std::cout << "load: " << (t_load_image - t0)/1000.f << "foward: " << (t_encode - t_load_image)/1000.f << "\n";
    ggml_free(model.ctx_w);

    return 0;
}


// python codes:
// features [N,OH,OW,C], pad_height= PH, check new height NH is the same with OH, if not, resize
// NH = ceil(OH/PH)*PH
// num_patch_heigh=num_pad_width = OH//PH
// num_patches=num_patch_heigh*num_pad_width
// features = transpose(features, [0, 3, 1,2] -> // [N, C,OH, OW] (1)
// patches = features.reshape(N*C*num_patch, PH, num_patch, PW] (2)
// patches = transpose(patches, [0, 2, 1, 3]  // (N*C*num_patch, num_patch, PH, PW) (3)
// patches = pathces.reshape(N, C, num_patches, patch_area) (4)
// patches = transpose(patch, [0, 3, 2, 1]) -> // (N, patch_area, num_patches, C))  (5)
// patches = patches.reshape(N*patch_area, num_patches, C) (6)

ggml_tensor * mobile_vit_layer::unfolding(
    ggml_context * ctx, ggml_tensor * features, int patch_size){
    // features has shape of (NW, NH, C, N) -> at step 1
    int nw = features->ne[0];
    int nh = features->ne[1];
    int c  = features->ne[2];
    int ps = patch_size;
    
    GGML_ASSERT(nw % ps == 0);
    GGML_ASSERT(nh % ps == 0);
    int n_patch_h = nh/ps;
    int n_patch_w = nw/ps;
    int n_patches = n_patch_h * n_patch_w;

    // step 2
    ggml_tensor * patches = ggml_reshape_4d(
        ctx, ggml_cont(ctx, features), ps, n_patch_w, ps, 1*c*n_patch_h
    ); 
    // step 3 [pw, n_patch_w, ph, c*n_patch_h*1] ->? [pw, ph, n_patch_w, 1*c*n_patch_h]
    patches = ggml_permute(ctx, patches, 0, 2, 1, 3);  // [pw, ph, n_patch_w, 1*c*n_patch_h]
    // step 4 resh[ae
    patches = ggml_reshape_4d(ctx, ggml_cont(ctx, patches), ps*ps, n_patch_w*n_patch_h, c, 1);
    // step 5 permute -> [c, n_patch_h*n_patch_w, ps*ps*1] // 3d matrix
    patches = ggml_cont(ctx, ggml_permute(ctx, patches, 2, 1, 0, 3));

    print_shape("patches: ", patches);
    return patches;
}


ggml_tensor * mobile_vit_layer::folding(ggml_context * ctx, ggml_tensor * features, int patch_size){
    // features (C, n_patches, patch_area*batch) 
    int c = features->ne[0];
    int n_patches = features->ne[1];
    int n_patch = (int) (sqrt(1.0*n_patches)); // assume image is a square, and patch is square
    int patch_area = patch_size*patch_size;

    features = ggml_reshape_4d(ctx, features, c, n_patches, patch_area, 1); // (c, n_patches, ps*ps,  1)
    features = ggml_cont(ctx, ggml_permute(ctx, features, 2, 1, 0, 3));  // (ps*ps, n_patches, c, batch)
    features = ggml_reshape_4d(ctx, features, patch_size, patch_size, n_patch, n_patch*c*1);
    features = ggml_cont(
        ctx,
        ggml_permute(ctx, features, 0, 2, 1, 3)
    ); // (patch_size, n_patch, patch_size, n_patch*c*1) 
    features = ggml_reshape_4d(ctx, features, patch_size*n_patch, patch_size*n_patch, c, 1);


    return features;
}


struct ggml_tensor * mobilevit_conv_layer::forward(
    ggml_context * ctx,
    ggml_tensor * input,
    int s,
    bool use_normalization,
    bool use_activation,
    bool depthwise
){
    std::cout <<"mobilevit_conv_layer.forward\n";
    print_shape("kernel: ", kernel);
    print_shape("input: ", input);

    int oc = kernel->ne[0];
    int ic = kernel->ne[1];
    int kw = kernel->ne[2];
    int kh = kernel->ne[3];
    int padding = (kw - 1) /2;
    struct ggml_tensor * output;

    if (depthwise){
        std::cout << "in depthwise\n";
        output = ggml_conv_depthwise_2d(
            ctx,
            ggml_cont_4d(
                ctx,
                ggml_permute(ctx, kernel, 3, 2, 0, 1), // (OC, IC, KW, KH) -> (KW, KH, IC, OC)
                kw, kh, ic, oc
            ),
            input, s, s, padding, padding, 1, 1 // s0, s1, p0, p1, d0, d1
        );
        std::cout << "done depthwise\n";
    } else{
        output = ggml_conv_2d(
            ctx,
            ggml_cont_4d(
                ctx,
                ggml_permute(ctx, kernel, 3, 2, 0, 1), // (OC, IC, KW, KH) -> (KW, KH, IC, OC)
                kw, kh, ic, oc
            ),
            input, s, s, padding, padding, 1, 1 // s0, s1, p0, p1, d0, d1
        );
    }

    print_shape("output: ", output);
    if (use_normalization){
        output = ggml_sub(
            ctx,
            output,
            ggml_repeat(
                ctx,
                ggml_cont_4d(
                    ctx, moving_mean, 1, 1, moving_mean->ne[0], 1
                ),
                output
            )
        );
        output = ggml_div(
            ctx,
            output,
            ggml_sqrt(
                ctx,
                ggml_repeat(
                    ctx,
                    ggml_cont_4d(ctx, moving_variance, 1, 1, moving_variance->ne[0], 1),
                    output
                )
            )
        );
        output = ggml_mul(
            ctx,
            output,
            ggml_repeat(ctx, ggml_cont_4d(ctx, gamma, 1, 1, gamma->ne[0], 1), output)
        );
        output = ggml_add(
            ctx,
            output, ggml_repeat(ctx, ggml_cont_4d(ctx, beta, 1, 1, beta->ne[0], 1), output)
        );
    }

    if (use_activation){
        output = ggml_silu(ctx, output);
    }
    return output;
}

ggml_tensor * inverted_residual_layer::forward(
    ggml_context * ctx, ggml_tensor * inp
){
    ggml_tensor * features = inp; 

    std::cout << "inverted_residual_layer\n"; 
    std::cout << "  before expanding\n";

    // forward signature: ctx, inpt, stride, use_normalization, use_activation,  depthwise
    features = expand_1x1.forward(ctx, features, 1, true, true, false); 
    std::cout << "  before conv3x3\n";
    features = conv_3x3.forward(ctx, features, strides, true, true, true);
    std::cout << "  before reducing\n";
    features = reduce_1x1.forward(ctx, features, 1, true, true, false);

    if (strides == 1 && in_channels == out_channels){
        features = ggml_add(ctx, features, inp);
    }
    return features;
}

void read_all_weights(mobilevit_model& model, std::ifstream &fin){
    // First, read all the weights
    while (true){
        total_weights += 1;
        int name_length, n_dims;
        bool is_f16 = false;
        // read name_length
        fin.read(reinterpret_cast<char *>(&name_length), sizeof(name_length));

        // read name
        std::string name(name_length, 0);
        fin.read(&name[0], name_length);
        std::cout << "name: ***" << name << "*** ";

        // if the string contstrain convolution, change the datatypes to f32
        if (name.find("convolution") != std::string::npos){
            std::cout << " I am here\n";
            is_f16 = true;
        }

        // read n_dims
        fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
        std::cout << "n_dims: " << n_dims << ". ";
        
        int dims[4] = {1, 1, 1, 1};
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
            if (is_f16){
                tensor = ggml_new_tensor_4d(ctx_w, GGML_TYPE_F16, dims[3], dims[2], dims[1], dims[0]);
            } else{
                tensor = ggml_new_tensor_4d(ctx_w, GGML_TYPE_F32, dims[3], dims[2], dims[1], dims[0]);
            }
        }else if (n_dims == 3){
            tensor = ggml_new_tensor_3d(ctx_w, GGML_TYPE_F32, dims[2], dims[1], dims[0]);
        }else if (n_dims == 2){
            tensor = ggml_new_tensor_2d(ctx_w, GGML_TYPE_F32, dims[1], dims[0]);
        }else if (n_dims == 1){
            tensor = ggml_new_tensor_1d(ctx_w, GGML_TYPE_F32, dims[0]);
        }


        int matrix_size = dims[0]*dims[1]*dims[2]*dims[3];
        std::vector<float> original_data(matrix_size);
        fin.read(
            reinterpret_cast<char *>(original_data.data()), matrix_size*sizeof(float)
        );

        // if ithe data is f16, convert, else, assign

        if (is_f16){
            std::vector<ggml_fp16_t> f16data (matrix_size);
            ggml_fp32_to_fp16_row(original_data.data(), f16data.data(), matrix_size);
            memcpy(tensor->data, f16data.data(), ggml_nbytes(tensor));
            std::cout << "original: "<< matrix_size*sizeof(float) << ", converted: " << ggml_nbytes(tensor) << "\n";
        }else{
            memcpy(tensor->data, original_data.data(), ggml_nbytes(tensor));
        }
        
        model.tensors[name] = tensor;

        if (fin.eof()) {std::cout << "done loading" << "\n"; break;}

    }
}
 
void print_shape(std::string name, ggml_tensor* tensor){
    std::cout << name << ": ";
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


ggml_tensor * mobilevit_transformer::forward(
    ggml_context * ctx, ggml_tensor * inp, float layer_norm_eps
){
    for (int i = 0; i < num_stages; i++){
        inp = layers[i].forward(ctx, inp, layer_norm_eps, num_head);
    }
    return inp;
}


ggml_tensor * transpose_for_score(ggml_context * ctx, ggml_tensor* inp, int num_head){
    int batch_size  = 1;
    int nc          = inp->ne[0];
    int n_patches   = inp->ne[1];
    int patch_area  = inp->ne[2];

    GGML_ASSERT(nc % num_head == 0);
    int attention_head_size = nc/num_head;
    inp = ggml_reshape_4d(ctx, inp, attention_head_size, num_head, n_patches, batch_size * patch_area);
    inp = ggml_permute(ctx, inp, 0, 2, 1, 3); 
    return inp;
}

ggml_tensor * mobilevit_transformer_layer::forward(
    ggml_context * ctx, ggml_tensor * hidden_states, float layer_norm_eps, int num_head
){
    // attention_output = self_attention(layernorm_before(hidden_states)    (1)
    // hidden_states = attention_output + hidden_states                     (2)
    // layer_output = layernorm_after(hidden_sizes)                         (3)
    // layer_output = intermediate(layer_output)                            (4)
    // layer_output = mobilevit_output(layer_output, hidden_states, training=training), (5)
    std::cout << "------------>transformer_layer_forward\n";
    print_shape("hidden_states: ", hidden_states);
    std::cout << hidden_states->nb[0] << "\n";
    int attention_head_size = hidden_states->ne[0]/num_head;

    float scale = sqrt(1.0f*attention_head_size);

    //layernorm before
    GGML_ASSERT(hidden_states->nb[0] == sizeof(float));
    ggml_tensor * attention_inp = ggml_add(
        ctx,
        ggml_mul(
            ctx,
            ggml_norm(ctx, hidden_states, layer_norm_eps),
            ggml_repeat(
                ctx,
                ggml_reshape_3d(ctx, lb_gamma, lb_gamma->ne[0], 1, 1),
                hidden_states
            )
        ),
        ggml_repeat(
            ctx, 
            ggml_reshape_3d(ctx, lb_beta, lb_beta->ne[0], 1, 1),
            hidden_states
        )
    );
    attention_inp = ggml_cont(ctx, attention_inp);

    // key_layer
    ggml_tensor * key_layer = ggml_mul_mat(ctx, attention_key_kernel, attention_inp);
    key_layer = ggml_add(
        ctx,
        key_layer,
        ggml_repeat(
            ctx,
            ggml_reshape_3d(ctx, attention_key_bias, attention_key_bias->ne[0], 1, 1),
            key_layer
        )
    );
    key_layer = transpose_for_score(ctx, key_layer, num_head);


    // value_layer
    ggml_tensor * value_layer = ggml_mul_mat(ctx, attention_value_kernel, attention_inp);
    value_layer = ggml_add(
        ctx,
        value_layer,
        ggml_repeat(
            ctx,
            ggml_reshape_3d(ctx, attention_value_bias, attention_value_bias->ne[0], 1, 1),
            value_layer 
        )
    );
    value_layer = transpose_for_score(ctx, value_layer, num_head);   

    // query
    ggml_tensor * query_layer = ggml_mul_mat(ctx, attention_query_kernel, attention_inp);
    query_layer = ggml_add(
        ctx,
        query_layer,
        ggml_repeat(
            ctx,
            ggml_reshape_3d(ctx, attention_query_bias, attention_query_bias->ne[0], 1, 1),
            query_layer 
        )
    );
    query_layer = transpose_for_score(ctx, query_layer, num_head);  // head_size,L, num_head, batch 

    //Take the dot product between "query" and "key" to get the raw attention scores
    ggml_tensor * attention_score = ggml_div(
        ctx,
        ggml_mul_mat(ctx,key_layer, query_layer),
        ggml_new_f32(ctx, scale)
    ); // L, L, num_head, batch

    attention_score = ggml_soft_max(ctx, attention_score); // [L, L, num_head, batch 

    //we need to transpose value_layer in order to be ggml_mul_mat 
    ggml_tensor * context_layer = ggml_mul_mat(
        ctx,
        ggml_cont(ctx, ggml_permute(ctx, value_layer, 1, 0, 2, 3)),
        attention_score
    ); // head_size, L, num_head, B 

    context_layer = ggml_permute(ctx, context_layer, 0, 2, 1, 3); // head_size, num_head, L, B
    context_layer = ggml_reshape_3d(
        ctx,
        ggml_cont(ctx, context_layer),
        hidden_states->ne[0], hidden_states->ne[1], hidden_states->ne[2]
    ); // hidden_size, L, B

    ggml_tensor * attention_output = ggml_mul_mat(ctx, attention_output_kernel, context_layer);
    attention_output = ggml_add(
        ctx,
        attention_output,
        ggml_repeat(
            ctx,
            ggml_reshape_3d(ctx, attention_output_bias, attention_output_bias->ne[0], 1, 1),
            attention_output 
        )
    ); // hidden_states, L, B 

    // step  2 hidden_states = hidden_states + attention_output
    hidden_states = ggml_add(ctx, hidden_states, attention_output);

    // step 3: layer_output = layernorm_after(hidden_sizes)                         (3)
    ggml_tensor * layer_output = ggml_add(
        ctx,
        ggml_mul(
            ctx,
            ggml_norm(ctx, hidden_states, layer_norm_eps),
            ggml_repeat(
                ctx,
                ggml_reshape_3d(ctx, la_gamma, la_gamma->ne[0], 1, 1),
                hidden_states
            )
        ),
        ggml_repeat(
            ctx, 
            ggml_reshape_3d(ctx, la_beta, la_beta->ne[0], 1, 1),
            hidden_states
        )
    );
    layer_output = ggml_cont(ctx, layer_output);
    print_shape("layernorm after: ", layer_output);
    // step 4:
    // layer_output = intermediate(layer_output)                            (4)
    layer_output = ggml_mul_mat(
        ctx,
        ggml_cont(ctx, ggml_transpose(ctx, intermediate_kernel)),
        layer_output
    ); 
    layer_output = ggml_add(
        ctx,
        layer_output,
        ggml_repeat(
            ctx,
            ggml_reshape_3d(ctx, intermediate_bias, intermediate_bias->ne[0], 1, 1),
            layer_output 
        )
    );
    print_shape("intermediate: ", layer_output);
    layer_output = ggml_silu(ctx, layer_output);

    // step 5:layer_output = mobilevit_output(layer_output, hidden_states, training=training), (5)
    // In the implementaiton, layer_output is named to hidden_states. hidden_states is name in to input_tensor
    layer_output = ggml_mul_mat(ctx,
        ggml_cont(ctx, ggml_transpose(ctx, output_kernel)),
        layer_output
    ); 
    layer_output = ggml_add(
        ctx,
        layer_output,
        ggml_repeat(
            ctx,
            ggml_reshape_3d(ctx, output_bias, output_bias->ne[0], 1, 1),
            layer_output 
        )
    );

    layer_output = ggml_add(
        ctx,
        layer_output,
        hidden_states
    );

    return layer_output;
}

ggml_tensor * mobile_vit_layer::forward(ggml_context * ctx, ggml_tensor * inp){
        ggml_tensor * features = downsampling_layer.forward(ctx, inp);
        ggml_tensor * residual = features;

        // forward signature: ctx, inpt, stride, use_normalization, use_activation,  depthwise
        features = conv_kxk.forward(ctx, features, 1, true, true, false);
        features = conv_1x1.forward(ctx, features, 1, false, false, false);

        // unfold the features to [OC, N_PATCHES, PATCH_AREA, 1]
        features = unfolding(ctx, features, patch_size);

        //transformer
        print_shape("features before folding: ", features);
        features = transformer.forward(ctx, features, layer_norm_esp);
        print_shape("features after folding: ", features);
        
        // layernorm  
        features = ggml_add(
            ctx,
            ggml_mul(
                ctx,
                ggml_norm(ctx, features, layer_norm_esp),
                ggml_repeat(
                    ctx,
                    ggml_reshape_3d(ctx, layernorm_alpha, layernorm_alpha->ne[0], 1, 1),
                    features 
                )
            ),
            ggml_repeat(
                ctx, 
                ggml_reshape_3d(ctx, layernorm_beta, layernorm_beta->ne[0], 1, 1),
                features 
            )
        );
        features = ggml_cont(ctx, features);
        print_shape("features after layernorm: ", features);
    
        // folding
        features = folding(ctx, features, patch_size);
        print_shape("features after folding: ", features);

        // forward signature: ctx, inpt, stride, use_normalization, use_activation,  depthwise
        features = conv_projection.forward(ctx, features, 1, true, true, false);
        print_shape("features after projection: ", features);

        features = fusion.forward(
            ctx,
            ggml_concat(ctx, residual, features),
            1, true, true, false
        );
        print_shape("features after fusion: ", features);
        return features;
    }


