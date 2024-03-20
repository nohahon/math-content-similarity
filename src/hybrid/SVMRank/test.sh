#!/bin/bash

printf "Running SVMrank classification\n"
./svm_rank_classify -v 2 ./data/test.dat ./data/model_trained.dat  >/dev/null
printf "Running evaluation script\n"

python eval_predictions.py
