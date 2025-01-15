from torch_py.MinDQNRobot import MinDQNRobot as TorchRobot  # PyTorch版本
import matplotlib.pyplot as plt
from Maze import Maze
from Runner import Runner
import time


class Robot(TorchRobot):

    def __init__(self, maze):
        """
        初始化 Robot 类
        :param maze:迷宫对象
        """
        super(Robot, self).__init__(maze)
        maze.set_reward(reward={
            "hit_wall": 10.0,
            "destination": -maze.maze_size ** 2.0*8,
            "default": 1.0,
        })
        self.epsilon = 0.1
        """开启金手指，获取全图视野"""
        self.memory.build_full_view(maze=maze)
        # 初始化后即开始训练
        self.loss_list = self.train()

    def train(self):
        loss_list = []
        batch_size = len(self.memory)

        start = time.time()
        # 训练，直到能走出这个迷宫
        while True:
            loss = self._learn(batch=batch_size)
            loss_list.append(loss)
            self.reset()
            for _ in range(self.maze.maze_size ** 2):
                a, r = self.test_update()
                if r == self.maze.reward["destination"]:
                    print('Training time: {:.2f} s'.format(
                        time.time() - start))
                    return loss_list

    def train_update(self):
        state = self.sense_state()
        action = self._choose_action(state)
        reward = self.maze.move_robot(action)

        return action, reward


"""  Deep Qlearning 算法相关参数： """
epoch = 10  # 训练轮数
maze_size = 11  # 迷宫size
training_per_epoch = int(maze_size * maze_size * 1.5)

""" 使用 DQN 算法训练 """

maze = Maze(maze_size=maze_size)
robot = Robot(maze)
runner = Runner(robot)
runner.run_training(epoch, training_per_epoch)


"""Test Robot"""
robot.reset()
for _ in range(maze.maze_size ** 2 - 1):
    a, r = robot.test_update()
    print("action:", a, "reward:", r)
    if r == maze.reward["destination"]:
        print("success")
        break

# 生成训练过程的gif图, 建议下载到本地查看；也可以注释该行代码，加快运行速度。
runner.generate_gif(filename="results/dqn_size10.gif")

# 绘制损失曲线
loss_list = robot.loss_list
n = len(loss_list)
plt.plot(range(n), loss_list)
plt.xlabel('Epochs')  # 可选：设置 x 轴标签
plt.ylabel('Loss')    # 可选：设置 y 轴标签
plt.title('Loss Curve')  # 可选：设置图表标题
plt.show()
