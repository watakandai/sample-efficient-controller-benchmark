import os
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
EXAMPLE_DIR = os.path.dirname(CURRENT_DIR)
HOME_DIR = os.path.dirname(EXAMPLE_DIR)
sys.path.append(HOME_DIR)

import logging
from pathlib import Path
from datetime import datetime
from sciab import util


if __name__ == '__main__':

    args = util.parse_args()

    datestr = datetime.today().strftime('%Y-%m-%d-%H:%M')
    filedir = os.path.join("output", f"{datestr}")
    Path(filedir).mkdir(parents=True, exist_ok=True)

    logging.basicConfig(filename=os.path.join(filedir, "experiment.log"),
                        level=logging.DEBUG,
                        format='%(asctime)s [%(levelname)s]: %(message)s')

    csv_filename = "benchmark.csv"
    column_labels = ['algorithm', 'env', '# iterations', 'time',
                     '# states', '# actions',
                     'initLB', 'initUB',
                     'termLB', 'termUB',
                     'safeLB', 'safeUB',
                     'controlLB', 'controlUB',
                     'dt']
    util.write_header(column_labels, csv_filename, print_outputs=True)
    f = open(csv_filename, 'a+')

    stats = util.selectModulesAndRunExperiment(args.pathToExecutable,
                                          args.algorithm,
                                          args.envName,
                                          args.numSample,
                                          filedir,
                                          args.plot)

    msg = ','.join([str(col) for col in stats])
    f.write(msg+'\n')

    f.close()
