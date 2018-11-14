#!/usr/bin/env python
"""
imports a set of experiments from experiments.py, runs them, and write results
"""
import experiments as exp
from utils import load_data, show_example, get_y_map, convert_y
import logging

import matplotlib.pyplot as plt
import numpy as np
import os

# adds a simple logger
logging.basicConfig(level=logging.INFO, format="[%(name)s:%(funcName)s:%(lineno)s] %(levelname)s: %(message)s")
LOGGER = logging.getLogger(os.path.basename(__file__))

category_map = {}

def main():

    log_fname = "logs/experiment01_log.txt"
    if os.path.isfile(log_fname):
        os.remove(log_fname)
    log_hdl = logging.FileHandler(log_fname)
    log_hdl.setFormatter(logging.Formatter('%(message)s'))
    LOGGER.addHandler(log_hdl)

    data = load_data()

    # way to map between string labels and int labels
    y_map = get_y_map(data)
    data['y']['train'] = convert_y(data['y']['train'], y_map)
    data['y']['valid'] = convert_y(data['y']['valid'], y_map)

    predictions, model = exp.experiment05(data) # svm

    import IPython; IPython.embed()
    # write results


if __name__ == "__main__":
    main()



