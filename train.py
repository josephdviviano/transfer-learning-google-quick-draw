"""
imports a set of experiments from experiments.py, runs them, and write results
"""
import os
import logging
from experiments import experiment01

# adds a simple logger
logging.basicConfig(level=logging.INFO, format="[%(name)s:%(funcName)s:%(lineno)s] %(levelname)s: %(message)s")
LOGGER = logging.getLogger(os.path.basename(__file__))


def main():

    log_fname = "experiment01_log.txt"
    if os.path.isfile(log_fname):
        os.remove(log_fname)
    log_hdl = logging.FileHandler(log_fname)
    log_hdl.setFormatter(logging.Formatter('%(message)s'))
    LOGGER.addHandler(log_hdl)

    # results = experiment01
    # write results
    pass


if __name__ == "__main__":
    main()


