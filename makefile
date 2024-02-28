OBJ=./objects

all: mkdir simple_rnn rnn_generation 

rnn_generation: ${OBJ}/ggml.o ${OBJ}/rnn_text_gen.o
	g++ -g -o rnn_generation_main ${OBJ}/rnn_text_gen.o  ${OBJ}/ggml.o -lm -lpthread

mkdir: 
	mkdir -p ${OBJ}

${OBJ}/ggml.o: ggml.c
	gcc -c -I. -o ${OBJ}/ggml.o ggml.c

${OBJ}/simple_rnn.o: simple_rnn/simple_rnn.cpp
	g++ -I. -o ${OBJ}/simple_rnn.o -c simple_rnn/simple_rnn.cpp


${OBJ}/rnn_text_gen.o: rnn_text_gen/rnn_text_generation.cpp
	g++ -I. -o ${OBJ}/rnn_text_gen.o -c rnn_text_gen/rnn_text_generation.cpp


.PHONY: clean


clean:
	rm -f ${OBJ}/*.o	
