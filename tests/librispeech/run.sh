#!/bin/bash

./run_eval.sh > log/total.log  2>&1
echo $?
./run_eval_r.sh >> log/total.log  2>&1
echo $?
echo "All done."