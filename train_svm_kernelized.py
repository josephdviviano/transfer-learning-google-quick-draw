#!/usr/bin/env python
"""
imports a set of experiments from experiments.py, runs them, and write results
"""
import utils
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

    #log_fname = "logs/train13.log"
    #if os.path.isfile(log_fname):
        #os.remove(log_fname)
    #log_hdl = logging.FileHandler(log_fname)
    #log_hdl.setFormatter(logging.Formatter('%(message)s'))
    #LOGGER.addHandler(log_hdl)

    #data = utils.load_data(test_mode=test_mode)
    #data = utils.load_data_2d(test_mode=test_mode)
    data = utils.load_data(test_mode=test_mode, cropping = True)

    # way to map between string labels and int labels
    y_map = utils.get_y_map(data)
    data['y']['train'] = utils.convert_y(data['y']['train'], y_map)
    data['y']['valid'] = utils.convert_y(data['y']['valid'], y_map)


    # run experiments
    svm_pred, svm_model = exp.svm_nonlinear(data)
    svm_y_test = utils.convert_y(svm_pred['test'], y_map)
    utils.write_results('results/svm_kernelized.csv', svm_y_test)
    

    
    
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


