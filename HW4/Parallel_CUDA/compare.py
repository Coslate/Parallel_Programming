#! /usr/bin/env python3
import subprocess
import sys
import time
import re

def ReadReturnData(input_file, total_data_list):
    with open(input_file, 'r') as f:
        read_serial = f.readlines()

    count_line = 0
    for line in read_serial:
        if(re.match(r'\s*Printing\s*final\s*results...\s*', line)):
            count_line = 1
            continue

        if((re.match(r'\s*Done\s*', line)) or (not(re.match(r'\S+', line)))):
            count_line = 0
            continue

        if(count_line):
            for x in line.split():
                if(x is not None):
                    total_data_list.append(float(x))

    total_data_list = [x for x in total_data_list if(x is not None)]

def CalculateErrorValue(in1_list, in2_list):
    if(len(in1_list) != len(in2_list)):
        print("Error: len(serial_data) does not equal to len(cuda_data).")
        exit(1)

    err_count = 0
    for i, x in enumerate(in1_list):
        error_value = abs(in1_list[i]-in2_list[i])
        if(error_value > 0.01):
            err_count += 1
            print("Error: read_serial_data - read_cuda_data => {} - {} = {} is larger than 0.01.".format(in1_list[i], in2_list[i], error_value))

    if(err_count == 0):
        print("PASS")

def RunCuda(p, s):
    f = open("./result_cuda.txt", "w")
    subprocess.call(["./cuda_wave", p, s], stdout=f)

def RunSerial(p, s):
    f = open("./result_serial.txt", "w")
    subprocess.call(["../Serial/serial_wave.o", p, s], stdout=f)

def TestRun(p, s):
    start_time1 = time.time()
    RunSerial(p, s)
    end_time1 = time.time()

    start_time2 = time.time()
    RunCuda(p, s)
    end_time2 = time.time()

    print("serial output diff with cuda output")
    RunDiff()
    print("")
    print("serial wave equation program run {} seconds".format(end_time1 - start_time1))
    print("cuda wave equation program run {} seconds".format(end_time2 - start_time2))

def RunDiff():
    subprocess.call(["diff", "result_cuda.txt", "result_serial.txt"])

    read_serial_data = []
    read_cuda_data   = []
    ReadReturnData("./result_serial.txt", read_serial_data)
    ReadReturnData("./result_cuda.txt", read_cuda_data)

#    print("read_serial_data : ")
#    for i, x in enumerate(read_serial_data):
#        print("{} ".format(x), end='')
#        if(i%10==0):
#            print("")
#
#    print("")
#    print("read_cuda_data : ")
#    for i, x in enumerate(read_cuda_data):
#        print("{} ".format(x), end='')
#        if(i%10==0):
#            print("")
#    print("")

    CalculateErrorValue(read_serial_data, read_cuda_data)

#########################
#     Main-Routine      #
#########################
def main():
    if len(sys.argv) != 3:
        print("Usage:\n\t ./compare.py tp ns")
        sys.exit()

    TestRun(sys.argv[1], sys.argv[2])

#-----------------Execution------------------#
if __name__ == '__main__':
    main()
