# inverse_kinematics_prediction
Solving the inverse kinematics problem using supervised learning.

## Setup
This repo uses a Conda environment configured in `environment.yml`. Here are the steps to set these up properly from this repos home folder:
1. Create an new Conda environment `conda env create -f environment.yml`
2. Activate the environment `conda activate Inverse-Kinematics-Prediction`

If changes are made to `environment.yml` then update by running `conda env update --file environment.yml --prune`

## TODO
- Create a graph represenation of the information in `dataset.csv`
    - Know how nodes and edges are represented and what are their weights/attributes
- How are nodes or graphs embedded?
- Develop a training architecture
- Imeplement an analytical method for solving IK in the RobotArm.py class
    - Jacobian Transpose
    - Pseudoinverse method
    - Cyclic Coordinate Descent
    - Gradient descent approaches
    - Particle swarm optimization