import argparse
from src.logger_setup import log
from src.cirq_test import contract_with_tensorflow
from src.cirq_test import get_parallel_contraction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('circuit_file', help='file with circuit')
    # parser.add_argument('target_state',
    #                     help='state x against which amplitude is computed')
    parser.add_argument('--with_quickbb',
                        dest='quickbb_command',
                        default='./quickbb',
                        help='path to quickbb executable')
    args = parser.parse_args()

    # contract_with_tensorflow(args.circuit_file, args.quickbb_command)
    get_parallel_contraction(args.circuit_file)

if __name__=="__main__":
    main()
