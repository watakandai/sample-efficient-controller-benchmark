# Stable-Controller Inference Algorithms Benchmark (SCIAB)


# OMPL
## Installation

### Prerequisite CASTXML
DO NOT INSTALL castxml using BREW!!!
https://github.com/ompl/ompl/issues/628

Build CastXML from source. https://github.com/CastXML/CastXML
```bash
mkdir build
cd build
cmake .. -DLLVM_DIR=/usr/local/Cellar/llvm/16.0.2/lib/cmake/llvm/ # I had to point to the correct directory
make
```

## Prerequisite Others
```bash
brew install eigen castxml numpy boost-python3 pypy3 flann boost-python
sudo -H pip3 -v install pygccxml
```

### OMPL (with Python Binding) Installtion
```bash
git clone https://github.com/ompl/ompl
cd ompl
mkdir -p build/Release
cd build/Release
cmake -DCASTXML=/usr/local/bin/castxml ../.. # the location depends on pc.
make -j 4 update_bindings   # installs python binding
make -j4
```

For debugging purpose
```bash
make VERBOSE=1 update_bindings
```


## Instead use C++ package I made (ControlLyapunovFunctionPlanners)
Must pass a path to the ompl executable.
```bash
--pathToExecutable /home/kandai/Documents/projects/dev/ControlLyapunovFunctionPlanners/build/DubinsCar
```

# Result

#### Voronoi:

2023-05-11 00:23:18,645 [INFO]: ompl/Car2D-v0: #Samples=4, ElapsedTime=107.58417391777039
2023-05-16 12:03:01,278 [INFO]: ompl/DubinsCar-v0: #Samples=33, ElapsedTime=1278.616891860962
2023-05-16 18:53:24,954 [INFO]: ompl/DubinsCarWithAcceleration-v0: #Samples=289, ElapsedTime=23736.698593616486
ompl/Unicycle-v0: #Samples>500 ....
CartPole 0 chance after 900 iterations...

#### RL:

algorithm,env,iterations,time
ompl/Car2D-v0,107,2156.3760409355164
ompl/DubinsCar-v0,307,1327.3937578201294
ompl/DubinsCarWithAcceleration-v0,163,596.0054759979248
ompl/Unicycle-v0,152,5401.479188919067
ompl/CartPole-v0,229,3082.3583188056946

Input Mean for RL::
Env: Car2D, successRate: 1.0, Mean: [0.6621944]
Env: DubinsCar, successRate: 1.0, Mean: [0.38691995]
Env: DubinsCarWithAcceleration, successRate: 1.0, Mean: [0.70344824 0.22804886]
Env: Unicycle, successRate: 1.0, Mean: [0.69036293 0.37582266]
Env: UnicycleWithConstraint, successRate: 1.0, Input Mean: [0.4992313  0.41291764]

Input Mean for OMPL:
Env: Car2D, Input Mean: [0.66497264]
Env: DubinsCar, Input Mean: [0.69977616]



# Current Goal is to use MPC (as a Trainer and an env?)

Env -> Trainer(Controller, Env) -> Verifier -> Sampler
