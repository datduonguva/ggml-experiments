OBJ=./objects


# TODO: I am here
all: mkdir simple_rnn test3 rnn_generation 

rnn_generation: ${OBJ}/ggml.o ${OBJ}/rnn_text_gen.o
	g++ -g -o rnn_generation_main ${OBJ}/rnn_text_gen.o  ${OBJ}/ggml.o -lm -lpthread

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

${OBJ}/rnn_text_gen.o: rnn_text_gen/rnn_text_generation.cpp
	g++ -I. -o ${OBJ}/rnn_text_gen.o -c rnn_text_gen/rnn_text_generation.cpp


.PHONY: clean


clean:
	rm -f ${OBJ}/*.o	
