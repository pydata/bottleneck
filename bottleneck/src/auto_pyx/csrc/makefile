
all: clean
	@gcc mm.c -lm -Wall -Wextra
	@./a.out

gdb: clean
	@gcc mm.c -lm -g -Wall -Wextra
	@gdb ./a.out

valgrind: clean
	@gcc mm.c -lm -g -Wall -Wextra
	@valgrind --tool=memcheck --leak-check=yes --show-reachable=yes \
     --num-callers=20 ./a.out

clean:
	@rm -rf a.out
