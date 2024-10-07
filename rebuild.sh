rm rm saliencyDetection/models/horus*
rm nohup.out
> logs/horus.log

nohup python3 run.py build --verbose staging &