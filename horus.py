# Author Federico Rosatelli
import time
import argparse
import logging.config
import traceback
import yaml


def arguments_parser(args:argparse.Namespace) -> None:
    assert not (args.build and args.test and args.run), f"Only one argument allowed!"
    if args.build and (args.build != "build" and args.build != "test" and args.build != "run"):
        raise ValueError(f"Unrecognized argument '{args.build}'")

    if not args.config:
        args.config = "config/conf.yaml"
    conf = getConfigYAML(args.config)
    
    logger = getLogger(args.verbose if args.verbose else "development")

    main(args,conf,logger)


def main(args:argparse.Namespace,conf:any,logger:logging.Logger) -> None:
    if args.version:
        print("Horus Version: %s" % conf["version"])
        return
    
    from saliencyDetection import saliency
    from tester import tests
    
    start_time = time.time()
    
    try:
        if args.build == "build":
            saliency.trainHorusNetwork(conf["saliencyDetection"],verbose=args.verbose)

        elif args.build == "test":
            conf = getConfigYAML("config/tests.yaml")
            tests.testerCommandControll(conf)
        
        elif args.build == "run":
            saliency.runHorus(conf["saliencyDetection"],"Human_00192.avi")
            return

    except Exception as e:
        print(traceback.format_exc())
        logger.fatal(f"{e.__class__.__name__}: {' | '.join(map(str, e.args))} - Total Time: %.2f s" % (time.time()-start_time))
        return
    
    except KeyboardInterrupt:
        logger.fatal(f"Interrupt by User - Total Time: %.2f s" % (time.time()-start_time))
        return
    
    logger.info(f"Total Time: %.2f s" % (time.time()-start_time))

    return
    
    
    


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

def createConfigYAML(conf_dict:dict,file_name:str) -> None:
    with open(file_name,'w') as wr:
        yaml.dump(conf_dict,wr)
    return




class Horus:
    model_conf_file:str = "config/conf.yaml"
    test_conf_file:str  = "config/tests.yaml"
    logger_file:str     = "config/logger.yaml"

    def __init__(self) -> None:
        self.model_config   = getConfigYAML(self.model_conf_file)
        self.test_config    = getConfigYAML(self.test_conf_file)
        pass

    def new_model_conf(self,conf_dict:dict,file_name:str):
        file_name = file_name + "new_conf.yaml"
        createConfigYAML(conf_dict,file_name)
        self.model_config = getConfigYAML(file_name)
        return
    
    def new_test_conf(self):
        return





if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='horus.py [argument]',
                    description='Horus Saliency Detection & Decision Maker.\n',
                    epilog='For more information do not hesitate to contact the developer on github or other platforms :)')
    
    parser.add_argument('build',nargs='?',help="Build the Horus Neural Network")
    parser.add_argument('test',nargs='?',help="Tester")
    parser.add_argument('run',nargs='?',help="Run Horus Neural Network")
    parser.add_argument('-c','--config',action='store',type=str,metavar='file',help="YAML config file name")
    parser.add_argument('--verbose',action='store',choices=["developer","staging","production"],help="Logger settings")
    parser.add_argument('-v','--version',action='store_true',help="Return Horus Version")
    args = parser.parse_args()
    arguments_parser(args)
