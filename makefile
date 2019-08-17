CC = g++

OBJ = model.o layers.o gpu.o util.o param.o label.o

HOST = ./host
INC_PATH = ./include

run:		$(HOST)/main.c $(OBJ) $(INC_PATH)/types.h $(INC_PATH)/main.h
			$(CC) $(OBJ) $(HOST)/main.c -o run -lOpenCL -I./include

model.o:	$(HOST)/model.c $(INC_PATH)/types.h $(INC_PATH)/main.h 
			$(CC) -c $(HOST)/model.c -lOpenCL -I./include

layers.o:	$(HOST)/layers.c $(INC_PATH)/types.h $(INC_PATH)/main.h
			$(CC) -c $(HOST)/layers.c -lOpenCL -I./inlcude

gpu.o:		$(HOST)/gpu.c $(INC_PATH)/main.h $(INC_PATH)/types.h 
			$(CC) -c $(HOST)/gpu.c -lOpenCL -I./include

util.o:		$(HOST)/util.c $(INC_PATH)/main.h $(INC_PATH)/types.h 
			$(CC) -c $(HOST)/util.c -lOpenCL -I./include

param.o:	$(HOST)/param.c $(INC_PATH)/main.h $(INC_PATH)/types.h
			$(CC) -c $(HOST)/param.c -lOpenCL -I./include

label.o:	$(HOST)/label.c $(INC_PATH)/main.h $(INC_PATH)/types.h
			$(CC) -c $(HOST)/label.c -lOpenCL -I./include

clean:		
			rm *.o
			rm run
