import argparse
import sys
import re
import csv
import numpy as np

def process(f):
    regex_tpt = re.compile('^(.*)Throughput:')
    regex_log = re.compile('^(.*).log')
    summary = []

    batchsize = 0
    throughputs = []
    for linenum, line in enumerate(f, 1):
        line = line.strip()
        if line == "":
            continue
        
        print(f"line {linenum} : [{line}]")
        
        m_tpt = re.match(regex_tpt, line)
        m_log = re.match(regex_log, line)
        if m_tpt is not None:
            num = line.split(":")[1].split("FPS")[0].strip()
            throughputs.append(float(num))
            print(f"{throughputs} ")
        if m_log is not None:
            bs = line.split(".")[0].split("_")[1].strip()
            mode = line.split(".")[0].split("_")[2].strip()
            batchsize = int(bs)
            print(f"{batchsize} {mode}")
            
        # reset
        if batchsize != 0 and len(throughputs) == 3:
            summary.append([batchsize] + throughputs)
            batchsize = 0
            throughputs = []
    
    return summary

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='log data processing, e.g. cat hint-tpt-112.txt |python collect_summary.py -o output.csv')
    parser.add_argument('-o', "--output", type=str, help='output csv', default="output.csv")
    parser.add_argument('-i', "--input", type=str, help='input log', default="")
    args = parser.parse_args()

    summary = []
    if args.input:
        with open(args.input, 'r', encoding = 'utf-8') as f:
            summary = process(f)
    else:
        summary = process(sys.stdin)
        
    with open(args.output, 'w', encoding = 'utf-8') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["BS", "bound", "static", "unbound"])
        
        def sortBS(val):
            return val[0]
        summary.sort(key=sortBS, reverse=False)
        
        print(summary)
        print(np.array(summary).shape)
        for r in summary:
            csv_writer.writerow(r)