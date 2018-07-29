from src.qOp import  *
from src.logging import log
import sys, re
OP = qOperation()

def main():
    f = sys.argv[1]
    log.info("reading file %s"%f)
    ops = []

    with open(f,"r") as cirquit:
        idx = 0
        qbit_count = 0
        for line in cirquit:
            if idx==0:
                qbit_count = int(line)
                log.info("There are %d qubits in cirquit"%qbit_count)
            else:
                m = re.search(r'([0-9]+) [a-z]', line)
                if not m:
                    raise Exception("file format error at line %d"%idx)
                layer_num = int(m.group(1))

                # +1 is for the space between layer idx and gate
                # -1 is for newline character
                op_str = line[len(m.group(1))+1 : -1]
                op = OP.factory(op_str)
                ops.append(op)
            idx +=1
    print("DONE\n",ops)

if __name__=="__main__":
    main()
