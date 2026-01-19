
move_copy_construction01.out : move_copy_construction01.o
	g++ -o move_copy_construction01.out move_copy_construction01.o

move_copy_construction01.o : move_copy_construction01.cpp
	g++ -c move_copy_construction01.cpp


.PHNOY: clean

clean:
	rm -rf *.o


