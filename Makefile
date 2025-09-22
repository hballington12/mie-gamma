FC = gfortran
FCFLAGS = -O3 -Wall -std=legacy

SRC = src/spher_f_mono.f
OBJ = spher_f_mono.o
TARGET = spher_f_mono

all: $(TARGET)

$(TARGET): $(OBJ)
	$(FC) -o $(TARGET) $(OBJ)

$(OBJ): $(SRC)
	$(FC) $(FCFLAGS) -c $(SRC)

clean:
	rm -f $(OBJ) $(TARGET)
