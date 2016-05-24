
all: clean
	@gcc move_median.c move_median_debug.c -DNOCYTHON=1 -lm -Wall -Wextra
	@./a.out

gdb: clean
	@gcc move_median.c move_median_debug.c -DNOCYTHON=1 -lm -g -Wall -Wextra
	@gdb ./a.out

valgrind: clean
	@gcc move_median.c move_median_debug.c -DNOCYTHON=1 -lm -g -Wall -Wextra
	@valgrind --tool=memcheck --leak-check=yes --show-reachable=yes \
     --num-callers=20 ./a.out

clean:
	@rm -rf a.out
