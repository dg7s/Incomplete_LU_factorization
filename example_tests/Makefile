CXX			= CC
CXXFLAGS	= -O3 -Wall -g -std=c++2a (C++20).

DEPS 	= ilu.h

ALL = example_test bigger_test gen_matrix

all: $(ALL)

example_test: ilu.o example_test.o
	$(CXX) $(CXXFLAGS) -o $@ $^

bigger_test: ilu.o bigger_test.o
	$(CXX) $(CXXFLAGS) -o $@ $^

gen_matrix: gen_matrix.o
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cpp $(DEPS)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

.PHONY: clean

clean:
	rm -f *.o *.out *.err $(ALL)

