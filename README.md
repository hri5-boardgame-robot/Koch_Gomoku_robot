# Koch_Gomoku_robot

## Overview
Koch_Gomoku_robot is a project that integrates a robotic arm with an AI-based Gomoku (Five in a Row) game. The robot can play Gomoku against a human player by physically placing and detecting stones on a board.

![Koch_Gomoku_robot Demo](imgs/demo.gif)  

[Watch the video demonstration](https://youtu.be/ioerYdkYhFU)  <!-- Add this line to include the YouTube link -->

## Features
- **AI-based Gomoku Player**: Uses Monte Carlo Tree Search (MCTS) and a neural network to play Gomoku.
- **Robotic Arm Control**: Controls a robotic arm to place stones on the board.
- **Vision System**: Detects the board and stones using a camera and computer vision techniques.
- **Simulation**: Uses MuJoCo for simulating the robotic arm.

## Project Structure
```
.
├── dynamixel.py
├── Gomoku_AI
│   ├── reference_model.py
│   ├── test_Vanilla_reference.py
│   └── Vanilla_MCTS
│       ├── game.py
│       ├── mcts_alphaZero.py
│       ├── policy_value_net.py
│       ├── renju_rule.py
│       ├── save
│       │   └── model_9
│       │       └── Vanilla_MCTS.model
│       ├── test.py
│       └── train_local.py
├── imgs
│   └── demo.gif
├── interface.py
├── LICENSE
├── low_cost_robot
│   ├── assets
│   │   ├── arm.stl
│   │   ├── base_link.stl
│   │   ├── base.stl
│   │   ├── elbow_to_wrist_extension.stl
│   │   ├── elbow-to-wrist-motor-reference_v1_1.stl
│   │   ├── elbow_to_wrist.stl
│   │   ├── elbow_to_wrist_with_motor.stl
│   │   ├── first-bracket-motor_1.stl
│   │   ├── Gomoku_board_1.STL
│   │   ├── Gomoku_board_2.STL
│   │   ├── Gomoku_piece_1.STL
│   │   ├── gripper-moving-part-dumb_v2_1.stl
│   │   ├── gripper_moving_part.stl
│   │   ├── gripper-static-motor-pt1.stl
│   │   ├── gripper-static-motor-pt2.stl
│   │   ├── gripper-static-motor_v2_1.stl
│   │   ├── gripper_static_part.stl
│   │   ├── shoulder_rotation.stl
│   │   ├── shoulder-to-elbow-motor_v1_1.stl
│   │   └── shoulder_to_elbow.stl
│   ├── low_cost_robot.xml
│   └── scene.xml
├── MJMODEL.TXT
├── omoku_bot_v2.py
├── play.py
├── README.md
├── requirements.txt
├── robot.py
├── sync_simul_real.py
├── tt.py
└── vision
    ├── initial.py
    ├── __init__.py
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
cd Gomoku_AI/Vanilla_MCTS
python test.py
```

### Training the Gomoku AI
To train the Gomoku AI, run:
```sh
cd ..
python Vanilla_MCTS/train_local.py
```

### Running the Robot
To run the robot and play Gomoku, run:
```sh
cd ..
python play.py
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


