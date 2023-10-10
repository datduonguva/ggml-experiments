OBJ=./objects


main: mkdir ${OBJ}/ggml.o  ${OBJ}/mnist_training.o
	g++ -o main ${OBJ}/mnist_training.o  ${OBJ}/ggml.o -lm -lpthread

mkdir: 
	mkdir -p ${OBJ}
${obj}/ggml.o: ../../src/ggml.c
	gcc -c -i../../include/ggml/ -o ${obj}/ggml.o ../../src/ggml.c

${OBJ}/mnist_training.o: train_mnist.cpp
	g++ -I../../include/ggml/ -o ${OBJ}/mnist_training.o -c train_mnist.cpp


