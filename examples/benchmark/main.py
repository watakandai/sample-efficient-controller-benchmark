import os
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
EXAMPLE_DIR = os.path.dirname(CURRENT_DIR)
HOME_DIR = os.path.dirname(EXAMPLE_DIR)
sys.path.append(HOME_DIR)

import logging
import itertools
from pathlib import Path
from datetime import datetime
from sciab import util


ALGORITHMS = ["RL", "randomVoronoi"]
# algorithms = ["ceVoronoi", "randomVoronoi"]
ENVNAMES = [
    "ompl/Car2D-v0",
    "ompl/DubinsCar-v0",
    "ompl/DubinsCarWithAcceleration-v0",
    "ompl/Unicycle-v0",
    "ompl/UnicycleWithConstraint-v0",
    # "ompl/CartPole-v0",
    # "ompl/Pendulum-v0",
    ## "ompl/Acrobot-v0",
    # "ompl/Drone2D-v0",
    ## "ompl/Drone3D-v0",
    ## "ompl/CaltechDuctedFan-v0"
]


if __name__ == '__main__':

    args = util.parse_benchmark_args()

    datestr = datetime.today().strftime('%Y-%m-%d-%H:%M')
    filedir = os.path.join("output", f"{datestr}")
    Path(filedir).mkdir(parents=True, exist_ok=True)

    logging.basicConfig(filename=os.path.join(filedir, "benchmark.log"),
                        level=logging.DEBUG,
                        format='%(asctime)s [%(levelname)s]: %(message)s')

    csv_filename = "benchmark.csv"
    util.write_header(csv_filename, print_outputs=True)
    f = open(csv_filename, 'a+')

    for (algorithm, envName) in itertools.product(ALGORITHMS, ENVNAMES):

        stats = util.selectModulesAndRunExperiment(args.pathToExecutable,
                                                   algorithm,
                                                   envName,
                                                   args.numSample,
                                                   filedir=None)

        msg = ','.join([str(col) for col in stats])
        f.write(msg + '\n')

    f.close()
