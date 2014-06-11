# include conf/platform.mk

CXX=g++-4.8
PYTHON=python2.7

SRCS = cexp.cpp

CFLAGS += -std=c++11 -Wall -Wno-conversion -Wno-unused-variable -Wno-return-type -O3 -ffast-math -fPIC
# CFLAGS += -std=c++11 -Wall -Wno-conversion -Wno-unused-variable -Wno-return-type -O0 -g -Wfatal-errors -fPIC
LIBS += -lm
INCLUDES= 

OBJS = $(SRCS:.cpp=.o)

test: libcexp.so
	$(CXX) $(CFLAGS) $(INCLUDES) -o bin/test test.cpp $(LFLAGS) $(LIBS) -L. -lcexp

.cpp.o:
	$(CXX) $(CFLAGS) $(INCLUDES) -c $<  -o $@

libcexp.so: $(OBJS)
	$(CXX) $(CFLAGS) $(INCLUDES) -o libcexp.so $(OBJS) $(LFLAGS) $(LIBS) -shared

clean:
	$(RM) order/*.o *.o *.so *~ $(MAIN)

depend: $(SRCS)
	makedepend $(INCLUDES) $^

python: libcexp.so
	CXX=$(CXX) $(PYTHON) setup.py build_ext -i

