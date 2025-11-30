# A3C (Asynchronous Advantage Actor-Critic)强化学习算法的复现和优化

下面是本人基于《Asynchronous Methods for Deep Reinforcement Learning》的论文实现的“A3C (Asynchronous Advantage Actor-Critic)强化学习算法的复现和优化”

## 1.配置环境

（1）激活conda环境

conda create --name a3c python=3.9

conda activate a3c

（2）安装需要的库

conda install -c conda-forge opencv

pip install gymnasium


conda install numpy

conda install matplotlib

conda install pytorch torchvision torchaudio cudatoolkit -c pytorch

(3)安装强化学习需要的环境

pip install gymnasium[atari]

pip install mujoco==2.3.3

pip install mujoco-py


## 2.代码解释

**（1）main.py**

在A3C中，有一个全局网络（global network）和多个工作智能体（worker），每个智能体都有自己的网络参数集。这些智能体中的每一个都与它自己的环境副本交互，同时其他智能体与它们的环境交互（并行训练）。这比单个智能体（除了加速完成更多工作）更好的原因在于，每个智能体的经验独立于其他智能体的经验。这样，可用于训练的整体经验多样化。

这个代码实现了一个 A3C (Asynchronous Advantage Actor-Critic) 强化学习算法的训练框架，利用多进程来并行训练多个智能体，以加速模型的学习过程。

a.代码通过 argparse 模块设置了多个训练参数，这些参数会在命令行中被传入程序：

--lr：学习率，默认值为 0.0001。

--gamma：折扣因子，用于奖励的折扣，默认值为 0.99。

--gae-lambda：用于 Generalized Advantage Estimation (GAE) 的 λ 参数，默认值为 1.00。

--entropy-coef：熵项的系数，用于在损失函数中控制策略的探索性，默认值为 0.01。

--value-loss-coef：价值函数损失的系数，默认值为 0.5。

--max-grad-norm：最大梯度规范，用于防止梯度爆炸，默认值为 50。

--seed：随机种子，确保实验的可复现性，默认值为 1。

--num-processes：并行训练进程的数量，默认值为 4。

--num-steps：A3C 算法中的每个训练步骤的步数，默认值为 20。

--max-episode-length：每个 episode 的最大长度，默认值为 1000000。

--env-name：训练的环境名称，默认值为 'PongDeterministic-v4'。

--no-shared：如果设置为 True，使用没有共享动量的优化器。

b.使用 create_atari_env 函数创建一个基于 Atari 游戏环境的强化学习环境。

**(2)model.py**

这段代码实现了 Actor-Critic 强化学习模型的定义，其中 Actor-Critic 模型结合了 值函数估计（Critic） 和 策略估计（Actor）。在这个模型中，Actor 负责生成动作的概率分布，Critic 负责估计当前状态的值。该模型特别适用于基于 深度强化学习（例如 A3C）的方法。

a.Actor-Critic 网络：该模型采用卷积层和 LSTM 层的组合来处理复杂的图像输入，卷积层负责特征提取，LSTM 层负责序列建模。

b.Critic：估算状态的价值，用于计算优势函数。

c.Actor：生成动作的概率分布，用于选择动作。

d.初始化：使用特定的初始化方法，确保网络训练时的稳定性。

**(3)envs.py**

实现了一个自定义的 OpenAI Gym 环境包装器，用于处理 Atari 游戏环境中的图像数据并对其进行缩放和归一化处理。它主要包括两个自定义的环境包装器：AtariRescale42x42 和 NormalizedEnv，并且提供了一个 create_atari_env 函数来创建该环境

a.AtariRescale42x42：用于将 Atari 游戏的图像缩放为 42x42 并转换为灰度图。

b.NormalizedEnv：用于对每个图像观测进行归一化，确保训练过程中每个观测的均值为 0，标准差为 1。

c.create_atari_env：创建并包装 Atari 环境，应用了上述两个包装器，确保图像处理和归一化。

**(4)train.py**

实现了多个进程的并行训练，其中每个进程都独立与环境交互，并通过 Actor-Critic 模型来计算梯度更新，并将结果同步到全局共享模型

**(5)test.py**

这段代码是强化学习中的 测试函数，主要用于 评估 一个训练好的 A3C（Asynchronous Advantage Actor-Critic）模型在 Atari 环境中的表现。

a.环境初始化：为每个进程创建并初始化一个 Atari 环境，并进行随机种子的设置以确保不同进程之间的环境一致性。

b.加载共享模型：在测试过程中，加载训练时共享的全局模型（shared_model），并使用该模型进行推理和决策。

c.环境交互：通过该模型与环境交互，得到每个动作的奖励，计算总奖励，记录 episode 的信息。

d.结果打印：每当一个 episode 完成时，打印时间、步骤、FPS、奖励等信息。


**(6)my_optim.py**

这段代码实现了一个自定义的 SharedAdam 优化器，它是 Adam 优化算法的变种，旨在 共享状态，以便在多进程或多线程的并行训练中同步模型参数的更新。具体来说，这个优化器可以在多个进程之间共享状态，使得它可以与 A3C（Asynchronous Advantage Actor-Critic）等并行训练算法一起使用。

特性：

a.共享优化器状态：每个进程都能访问和更新全局共享的优化器状态。

b.Adam 更新规则：根据经典的 Adam 优化算法，利用一阶矩和二阶矩进行参数更新，并通过动量和梯度的平方进行调整。

c.多进程支持：使用 share_memory_() 将优化器状态在不同进程之间共享，确保不同进程使用相同的优化器状态。

## 运行命令

python main.py --env-name "CartPole-v1" --num-processes 4
















