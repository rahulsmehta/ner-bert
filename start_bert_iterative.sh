#!/bin/bash
nohup python lt-iterative-bert.py > lt-iterative-bert.out 2> lt-iterative-bert.err &
echo $! > lt-iterative-bert.pid
