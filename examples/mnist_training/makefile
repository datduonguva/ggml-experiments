OBJ=./objects


main: mkdir ${OBJ}/ggml.o  ${OBJ}/mnist_training.o
	g++ -o main ${OBJ}/mnist_training.o  ${OBJ}/ggml.o -lm -lpthread

mkdir: 
	mkdir -p ${OBJ}
${OBJ}/ggml.o: ../../src/ggml.c
	gcc -c -I../../include/ggml/ -o ${OBJ}/ggml.o ../../src/ggml.c

${OBJ}/mnist_training.o: train_mnist.cpp
	g++ -I../../include/ggml/ -o ${OBJ}/mnist_training.o -c train_mnist.cpp


