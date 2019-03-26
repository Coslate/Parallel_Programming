#declare variable
CC = g++
INCLUDE_FILES = ./include
INCLUDE_FLAGS = $(foreach d, $(INCLUDE_FILES), -I $d)
CTAGS_UTIL = /usr/bin/
CTAGS_FILES = ./include
CTAGS_FLAGS = $(foreach d, $(CTAGS_FILES),-a $d/*)
CFLAGS = -g -Wall -O2 -std=c++11 -pthread -s
COMPILE_FLAGS = -c
MAIN_OBJECT_NAME_MAIN = pi
MAIN_OBJECT_SOURCE_MAIN = $(MAIN_OBJECT_NAME_MAIN).cpp
MAIN_OBJECT_EXEC_MAIN = $(MAIN_OBJECT_NAME_MAIN).o
MAIN_OBJECT_EXEC_OUTPUT = ./
MAIN_TESTING_FOLDER     = ./Testing/

all : clean all_file_tags
	$(CC) $(CFLAGS) $(INCLUDE_FLAGS) $(MAIN_OBJECT_SOURCE_MAIN) -o $(MAIN_OBJECT_EXEC_OUTPUT)$(MAIN_OBJECT_EXEC_MAIN)
	@time ${MAIN_OBJECT_EXEC_OUTPUT}$(MAIN_OBJECT_EXEC_MAIN) $(core) $(n)

all_file_tags : 
	$(CTAGS_UTIL)ctags -R ./*
	$(CTAGS_UTIL)ctags $(CTAGS_FLAGS)

clean : initialize
	rm -rf $(MAIN_OBJECT_EXEC_OUTPUT)$(MAIN_OBJECT_EXEC_MAIN)
	rm -rf ./tags

initialize : 
	for folder in $(INCLUDE_FILES) $(MAIN_TESTING_FOLDER) ; do \
		if [ ! -d $$folder ]; \
		then \
			mkdir $$folder; \
		fi \
	done
	cp -rf $(MAIN_OBJECT_SOURCE_MAIN) $(MAIN_TESTING_FOLDER) 
