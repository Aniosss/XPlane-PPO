from time import sleep
import xpc
from PPO import PPO
from PPO import gym_env
from PPO.NN import NeuralNetwork


def ex():
    print("X-Plane Connect example script")
    print("Setting up simulation")
    with xpc.XPlaneConnect() as client:
        env = gym_env.Env()

        ppo = PPO.PPO(NeuralNetwork, env)
        ppo.train_critic(20)
        ppo.train_actor(20, 5, 0.2)


if __name__ == "__main__":
    ex()
