# from dataLoader import dataLoader
# from evaluation import evaluator
#from objectDetection import visDrone


import torch
from saliencyDetection import saliency

# saliency.trainHorus(12)


import argparse
from tester import tests

#tests.unitTestCollider()

def arguments_parser(args:argparse.Namespace):
    if args.build:
        assert args.build > 0 and args.build <= 4096, "Value for build must be > 0 & < 4097"
        saliency.trainHorus(args.build)
        return
    if args.test:
        tests.unitTestCollider()
        return



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='Horus Saliency Detection & Decision Maker',
                    description='This program builds and uses the "Horus" neural network',
                    epilog='For more information do not hesitate to contact the developer on github or other platforms :)')
    
    parser.add_argument('-b','--build',action='store',type=int,metavar='n',help="Build the Horus Neural Network with n epoch")
    parser.add_argument('-t','--test',action='store_true',help="Tester")
    args = parser.parse_args()
    arguments_parser(args)
