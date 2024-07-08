# from dataLoader import dataLoader
# from evaluation import evaluator
#from objectDetection import visDrone



from saliencyDetection import saliency
import time
# saliency.trainHorus(12)


import argparse
from tester import tests
import logging.config
import yaml


def arguments_parser(args:argparse.Namespace):
    if not args.config:
        args.config = "config/conf.yaml"
    conf = getConfigYAML(args.config)
    
    logger = getLogger(args.verbose if args.verbose else "staging")

    main(args,conf,logger)

    

def main(args:argparse.Namespace,conf:any,logger:logging.Logger) -> None:
    start_time = time.time()
    #try:
    if args.build:
        saliency.trainHorus(conf["saliencyDetection"],verbose=args.verbose)

    elif args.test:
        tests.unitTestCollider()
    logger.info(f"Total Time: %.2f s" % (time.time()-start_time))

    # except Exception as e:
    #     logger.fatal(f"{e.__class__.__name__}: {' | '.join(e.args)} - Total Time: %.2f s" % (time.time()-start_time))
    #     return
    
    # except KeyboardInterrupt:
    #     logger.fatal(f"Interrupt by User - Total Time: %.2f s" % (time.time()-start_time))
    #     return
    
    
    


def getConfigYAML(conf_file:str) -> any:
    with open(conf_file, 'rt') as f:
        config = yaml.safe_load(f.read())
    return config

def getLogger(name:str) -> logging.Logger:
    with open('config/logger.yaml', 'rt') as f:
        config = yaml.safe_load(f.read())

    logging.config.dictConfig(config)

    logger = logging.getLogger(name)
    return logger




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='Horus Saliency Detection & Decision Maker',
                    description='This program builds and uses the "Horus" neural network',
                    epilog='For more information do not hesitate to contact the developer on github or other platforms :)')
    
    parser.add_argument('-b','--build',action='store_true',help="Build the Horus Neural Network")
    parser.add_argument('-t','--test',action='store_true',help="Tester")
    parser.add_argument('-c','--config',action='store',type=str,metavar='f',help="YAML config file name")
    parser.add_argument('-v','--verbose',action='store',choices=["developer","staging","production"],help="Logger settings")
    args = parser.parse_args()
    arguments_parser(args)
