# GGML Examples
Examples of inference of popular networks like DNN, RNN, LSTM, etc implemented in GGML.

## Models
- RNN Text Generation: Model is trained with Tensorflow, then exported to binary file. A GRU cell is implemented with GGML with the graph contains only the single GRU Cell. A simple for-loop is used to generate token by token.
- MobileVit: This is the conversion of huggingface.co's TFMobileViT to GGML.
