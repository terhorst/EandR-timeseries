all:
	python setup.py build_ext -i
clean:
	rm -f *.so *.o threelocus/*.o threelocus/*.so
