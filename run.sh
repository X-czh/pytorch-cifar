#!/bin/bash
# Shell scipt for autorun
# run "chmod 755 run.sh" first to make it executable

echo "Running.."

python main.py --model mlp --epoch 30
python main.py --model lenet --epoch 30

# TEST
#python main.py --model lenet --epoch 1

echo "Finished!"
