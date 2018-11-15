#!/usr/bin/env python
"""
imports a set of experiments from experiments.py, runs them, and write results
"""
from utils import load_data, show_example, get_y_map, convert_y, write_results
import argparse
import experiments as exp
import logging
import matplotlib.pyplot as plt
import numpy as np
import os

# adds a simple logger
logging.basicConfig(level=logging.INFO, format="[%(name)s:%(funcName)s:%(lineno)s] %(levelname)s: %(message)s")
LOGGER = logging.getLogger(os.path.basename(__file__))

category_map = {}

def main(test_mode=False):

    log_fname = "logs/train.log"
    if os.path.isfile(log_fname):
        os.remove(log_fname)
    log_hdl = logging.FileHandler(log_fname)
    log_hdl.setFormatter(logging.Formatter('%(message)s'))
    LOGGER.addHandler(log_hdl)

    data = load_data(test_mode=test_mode)

    # way to map between string labels and int labels
    y_map = get_y_map(data)
    data['y']['train'] = convert_y(data['y']['train'], y_map)
    data['y']['valid'] = convert_y(data['y']['valid'], y_map)

    # run experiments
    lr_pred, lr_model = exp.lr_baseline(data)
    lr_y_test = convert_y(lr_pred['test'], y_map)
    write_results('results/lr_baseline.csv', lr_y_test)

    svm_pred, svm_model = exp.svm_baseline(data)
    svm_y_test = convert_y(svm_pred['test'], y_map)
    write_results('results/svm_baseline.csv', svm_y_test)



if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument("-v", "--verbose", action="store_true",
        help="increase output verbosity")
    argparser.add_argument("-t", "--test", action="store_true",
        help="training set size=500")
    args = argparser.parse_args()

    if args.verbose:
        LOGGER.setLevel(logging.DEBUG)
    else:
        LOGGER.setLevel(logging.INFO)

    main(test_mode=args.test)


