#declare variable
CC = mpicc
INCLUDE_FILES =
INCLUDE_FLAGS = $(foreach d, $(INCLUDE_FILES), -I $d)
CTAGS_UTIL = /usr/share/zsh/functions/Completion/Unix/_ctags_tags
CTAGS_FILES = ./*
CTAGS_FLAGS = $(foreach d, $(CTAGS_FILES),-a $d/*)
CFLAGS =
COMPILE_FLAGS = -c
MAIN_OBJECT_NAME_MAIN = conduction
MAIN_OBJECT_SOURCE_MAIN = $(MAIN_OBJECT_NAME_MAIN).c
MAIN_OBJECT_EXEC_MAIN = $(MAIN_OBJECT_NAME_MAIN).o
MAIN_OBJECT_EXEC_OUTPUT = ./


all : clean
	$(CC) $(CFLAGS) $(INCLUDE_FLAGS) $(MAIN_OBJECT_SOURCE_MAIN) -o $(MAIN_OBJECT_EXEC_OUTPUT)$(MAIN_OBJECT_EXEC_MAIN)

test_pure: 
	$(MAIN_OBJECT_EXEC_OUTPUT)$(MAIN_OBJECT_EXEC_MAIN) $(N) $(seed)
test: 
	/usr/bin/time -f "%e" $(MAIN_OBJECT_EXEC_OUTPUT)$(MAIN_OBJECT_EXEC_MAIN) $(N) $(seed)

#all_file_tags : 
#       $(CTAGS_UTIL) -R ./*
#       $(CTAGS_UTIL) $(CTAGS_FLAGS)

clean :
	rm -rf $(MAIN_OBJECT_EXEC_OUTPUT)$(MAIN_OBJECT_EXEC_MAIN)

