from omoku_bot_v2 import OmokuBot
import time

TAKE_PICK_POSITION = [-0.08, 0.085, 0.14]


class TestOmokuBot:
    def __init__(self):
        self.robot = OmokuBot(use_real_robot=True)

    def run(self):
        while True:
            self.robot.move_ee_position_cartesian(TAKE_PICK_POSITION)
            self.robot.reload()

            self.robot.move_to_grid(0, 4)

            self.robot.move_down()

            self.robot.release_half()

            self.robot.move_up()
            # Add more commands as needed


if __name__ == "__main__":
    test = TestOmokuBot()
    test.run()
