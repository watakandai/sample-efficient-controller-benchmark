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
