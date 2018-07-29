from src.qOp import  *
from src.logging import log
import sys

def main():
    f = sys.argv[1]
    log.info("reading file %s"%f)

    with open(f,"r") as cirquit:
        idx = 0
        qbit_count = 0
        for line in cirquit:
            if idx==0:
                qbit_count = int(line)
                log.info("There are %d qubits in cirquit"%qbit_count)
            else:
                op = qOperation(line)
                print(op)
            idx +=1

if __name__=="__main__":
    main()
