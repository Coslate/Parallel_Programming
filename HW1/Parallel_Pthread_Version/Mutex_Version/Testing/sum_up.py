#! /usr/bin/env python3
import re
import argparse

#########################
#     Main-Routine      #
#########################
def main():
#Argument Parser
    (total_run) = ArgumentParser()

#Calculate
    one_core_time  = 0
    two_core_time  = 0
    four_core_time = 0
    final_score    = 0

    with open('./test_run.log') as f:
        lines = f.read().splitlines()

    for line in lines:
        m1      = re.match(r'\s*1\s*core\:\s*(\S+)\s*sec\s*', line)
        m2      = re.match(r'\s*2\s*core\:\s*(\S+)\s*sec\s*', line)
        m4      = re.match(r'\s*4\s*core\:\s*(\S+)\s*sec\s*', line)
        score_m = re.match(r'\s*Your\s*score\:\s*(\d+)\s*/\s*85\s*', line)
        if(m1):
            one_core_time += float(m1.group(1))

        if(m2):
            two_core_time += float(m2.group(1))

        if(m4):
            four_core_time += float(m4.group(1))

        if(score_m):
            final_score    += int(score_m.group(1))

    one_core_time  /= float(total_run)
    two_core_time  /= float(total_run)
    four_core_time /= float(total_run)
    final_score    /= float(total_run)

    print("============Summary============")
    print("1 core time = {}".format(one_core_time))
    print("2 core time = {}".format(two_core_time))
    print("4 core time = {}".format(four_core_time))
    print("final score = {}".format(final_score))

#########################
#     Sub-Routine       #
#########################
def ArgumentParser():
    total_run        = 0

    parser = argparse.ArgumentParser()
    parser.add_argument("--total_run_num", "-total_run", help="The number of the total run.")
    args = parser.parse_args()

    if args.total_run_num:
        total_run = int(args.total_run_num)

    return(total_run)

#-----------------Execution------------------#
if __name__ == '__main__':
    main()
