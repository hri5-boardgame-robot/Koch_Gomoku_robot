# Koch_Gomoku_robot

## Overview
Koch_Gomoku_robot is a project that integrates a robotic arm with an AI-based Gomoku (Five in a Row) game. The robot can play Gomoku against a human player by physically placing and detecting stones on a board.

![Koch_Gomoku_robot Demo](imgs/demo.gif)  

## Features
- **AI-based Gomoku Player**: Uses Monte Carlo Tree Search (MCTS) and a neural network to play Gomoku.
- **Robotic Arm Control**: Controls a robotic arm to place stones on the board.
- **Vision System**: Detects the board and stones using a camera and computer vision techniques.
- **Simulation**: Uses MuJoCo for simulating the robotic arm.

## Project Structure
```
.
├── dynamixel.py
├── Gomoku_AI/
│   ├── game.py
│   ├── LICENSE
│   ├── mcts_alphaZero.py
│   ├── policy_value_net.py
│   ├── README.md
│   ├── renju_rule.py
│   ├── save/
│   │   └── model_9/
│   ├── test.py
│   └── train_local.py
├── imgs/
├── interface.py
├── LICENSE
├── low_cost_robot/
│   ├── assets/
│   ├── low_cost_robot.xml
│   └── scene.xml
├── omoku_bot_v2.py
├── README.md
├── robot.py
├── sync_simul_real.py
├── test_robot.py
├── tt.py
├── vis.py
└── vision/
    ├── __init__.py
    ├── initial.py
    ├── test_camera.py
    ├── test_vision.py
    └── utils.py
```

## Installation
1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/Koch_Gomoku_robot.git
    cd Koch_Gomoku_robot
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

3. Install MuJoCo and set up the environment:
    Follow the instructions on the [MuJoCo website](https://mujoco.org/) to install MuJoCo and set up the necessary environment variables.

## Usage
### Running the Gomoku AI
To test the Gomoku AI, run:
```sh
python Gomoku_AI/test.py
```

### Training the Gomoku AI
To train the Gomoku AI, run:
```sh
python Gomoku_AI/train_local.py
```

### Running the Robot
To run the robot and play Gomoku, run:
```sh
python test_robot.py
```

### Vision System
To test the vision system, run:
```sh
python vision/test_camera.py
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements
- [Dynamixel SDK](https://github.com/ROBOTIS-GIT/DynamixelSDK)
- [MuJoCo](https://mujoco.org/)
- [PyTorch](https://pytorch.org/)


## Contact
For any questions or suggestions, please contact [your email](mailto:youremail@example.com).