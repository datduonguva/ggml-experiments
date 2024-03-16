"""
This script load the TFMobileViTModel model, the export all the weights to binary file
The weights are written in this order:

name_length,name,size_length,size_1,size_2,...,size_n,flatten_weights
"""
from transformers import TFMobileViTModel

model = TFMobileViTModel.from_pretrained("apple/mobilevit-small")

# for each layer, write: (name_length,name,size_length,size_1,size2,...,size_n,flatten_weights)
# this function works as expected
# TODO: convolution kernel should be stored as float16 before exporting
# TODO: DNN kernel should be permuted before exporting

with open("weight.ggml", "wb") as f:
    for weight in model.weights:
        print(weight.name, weight.shape)
        weight_name = weight.name
        weight_shape = weight.shape

        # write name_length and length
        f.write(struct.pack("i", len(weight_name)))
        f.write(bytes(weight_name, 'ascii'))


        # write n_dim, and values of each dim
        f.write(struct.pack("i", len(weight_shape)))
        for j in weight_shape:
            f.write(struct.pack("i", j))

        # write the weights
        weight.numpy().astype(np.float32).tofile(f)
