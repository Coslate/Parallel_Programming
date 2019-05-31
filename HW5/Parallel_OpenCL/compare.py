#! /usr/bin/env python3
import subprocess
import sys
import time
import re

def ReadReturnData(input_file, r_list, g_list, b_list):
    with open(input_file, 'r') as f:
        read_serial = f.readlines()

    count_line = 0
    is_r = 0
    is_g = 0
    is_b = 0
    for line in read_serial:
        if(re.match(r'\s*(R|G|B) = \s*', line)):
            count_line = 1
            if(re.match(r'\s*R = \s*', line)):
                is_r = 1
            if(re.match(r'\s*G = \s*', line)):
                is_g = 1
            if(re.match(r'\s*B = \s*', line)):
                is_b = 1
            continue

        if((re.match(r'\s*Done\s*', line)) or (not(re.match(r'\S+', line)))):
            count_line = 0
            if(is_r == 1):
                is_r = 0
            if(is_g == 1):
                is_g = 0
            if(is_b == 1):
                is_b = 0
            continue

        if(count_line):
            if(is_r):
                for x in line.split():
                    if(x is not None):
                        r_list.append(float(x))
            elif(is_g):
                for x in line.split():
                    if(x is not None):
                        g_list.append(float(x))
            elif(is_b):
                for x in line.split():
                    if(x is not None):
                        b_list.append(float(x))

    r_list = [x for x in r_list if(x is not None)]
    g_list = [x for x in g_list if(x is not None)]
    b_list = [x for x in b_list if(x is not None)]

def CalculateErrorValue(in1_list, in2_list, message_channel):
    if(len(in1_list) != len(in2_list)):
        print("Error: {} len(serial_data) does not equal to len(opencl_data).".format(message_channel))
        exit(1)

    err_count = 0
    for i, x in enumerate(in1_list):
        error_value = abs(in1_list[i]-in2_list[i])
        if(error_value >= 0.01):
            err_count += 1
            print("Error: {} {}-th data, read_serial_data - read_opencl_data => {} - {} = {} is larger than 0.01.".format(message_channel, (i+1), in1_list[i], in2_list[i], error_value))

    if(err_count == 0):
        print("PASS")

def RunParallel(bp):
    f = open("./result_opencl.txt", "w")
    subprocess.call(["./histogram", bp], stdout=f)

def RunSerial(bp):
    f = open("./result_serial.txt", "w")
    subprocess.call(["../Serial/histogram", bp], stdout=f)

def TestRun(bp):
    start_time1 = time.time()
    RunSerial(bp)
    end_time1 = time.time()

    start_time2 = time.time()
    RunParallel(bp)
    end_time2 = time.time()

    print("serial output diff with opencl output")
    RunDiff()
    print("")
    print("serial histogram program run {} seconds".format(end_time1 - start_time1))
    print("opencl histogram program run {} seconds".format(end_time2 - start_time2))

def RunDiff():
    f = open("./result_diff.txt", "w")
    subprocess.call(["diff", "result_opencl.txt", "result_serial.txt"], stdout=f)

    read_serial_r_data = []
    read_serial_g_data = []
    read_serial_b_data = []
    read_opencl_r_data = []
    read_opencl_g_data = []
    read_opencl_b_data = []
    ReadReturnData("./result_serial.txt", read_serial_r_data, read_serial_g_data, read_serial_b_data)
    ReadReturnData("./result_opencl.txt", read_opencl_r_data, read_opencl_g_data, read_opencl_b_data)

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

    CalculateErrorValue(read_serial_r_data, read_opencl_r_data, "R_CHANNEL")
    CalculateErrorValue(read_serial_g_data, read_opencl_g_data, "G_CHANNEL")
    CalculateErrorValue(read_serial_b_data, read_opencl_b_data, "B_CHANNEL")

#########################
#     Main-Routine      #
#########################
def main():
    if len(sys.argv) != 2:
        print("Usage:\n\t ./compare.py bp")
        sys.exit()

    TestRun(sys.argv[1])

#-----------------Execution------------------#
if __name__ == '__main__':
    main()
