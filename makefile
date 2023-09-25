OBJ=./objects


all: mkdir simple_rnn test3

simple_rnn: ${OBJ}/ggml.o  ${OBJ}/simple_rnn.o
	g++ -o simple_rnn_main ${OBJ}/simple_rnn.o  ${OBJ}/ggml.o -lm -lpthread

test3: mkdir ${OBJ}/ggml.o ${OBJ}/test3.o
	g++ -o test3_main ${OBJ}/test3.o ${OBJ}/ggml.o -lm -pthread

mkdir: 
	mkdir -p ${OBJ}

${OBJ}/ggml.o: ggml.c
	gcc -c -I. -o ${OBJ}/ggml.o ggml.c

${OBJ}/simple_rnn.o: simple_rnn/simple_rnn.cpp
	g++ -I. -o ${OBJ}/simple_rnn.o -c simple_rnn/simple_rnn.cpp

${OBJ}/test3.o: tests/test-mul-mat3.cpp
	g++ -I. -o ${OBJ}/test3.o -c tests/test-mul-mat3.cpp

.PHONY: clean

clean:
	rm -f ${OBJ}/*.o	
