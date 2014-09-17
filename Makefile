CXX=g++
CXXFLAGS=-std=c++0x -O3 -fopenmp -lz -I. -DKENLM_MAX_ORDER=6
#CXXFLAGS=-std=c++0x -g -fopenmp -lz -I. -DKENLM_MAX_ORDER=6
objs=lm/*.o util/*.o util/double-conversion/*.o

all: translator
translator: main.o translator.o lm.o ruletable.o vocab.o cand.o myutils.o syntaxtree.o $(objs)
	$(CXX) -o t2s main.o translator.o lm.o ruletable.o vocab.o myutils.o cand.o syntaxtree.o $(objs) $(CXXFLAGS)

main.o: translator.h stdafx.h cand.h vocab.h ruletable.h lm.h myutils.h syntaxtree.h
translator.o: translator.h stdafx.h cand.h vocab.h ruletable.h lm.h myutils.h syntaxtree.h
syntaxtree.o: syntaxtree.h cand.h myutils.h
lm.o: lm.h stdafx.h
ruletable.o: ruletable.h stdafx.h cand.h
vocab.o: vocab.h stdafx.h
cand.o: cand.h stdafx.h
myutils.o: myutils.h stdafx.h

clean:
	rm *.o
