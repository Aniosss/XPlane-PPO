import os.path
import time
from datetime import datetime

import numpy as np

import xpc
from PPO import PPO, gym_env

def scale_actions_to_correct(actions):
    actions[3] = actions[3] - 1
    actions[4] = max(actions[4], 0)
    return actions

def ex():
    print("X-Plane Connect example script")
    print("Setting up simulation")
    with xpc.XPlaneConnect() as client:
        # init hyperparams
        K_epochs = 80  # update policy for K epochs in one PPO update

        eps_clip = 0.2  # clip parameter for PPO
        gamma = 0.99  # discount factor

        lr_actor = 0.0003  # learning rate for actor network
        lr_critic = 0.001  # learning rate for critic network

        print_freq = 10000
        log_freq = 2000
        save_model_freq = int(1e5)

        action_std = 0.6  # starting std for action distribution (Multivariate Normal)
        action_std_decay_rate = 0.05  # linearly decay action_std (action_std = action_std - action_std_decay_rate)
        min_action_std = 0.1  # minimum action_std (stop decay after action_std <= min_action_std)
        action_std_decay_freq = int(2.5e5)

        # init enviroment params
        env_name = "XPlane"
        env = gym_env.Env()
        ppo_agent = PPO.PPO(env, lr_actor, lr_critic, gamma, num_epochs=K_epochs, eps_clip=eps_clip)

        # log files
        log_dir = "PPO_logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        log_dir = log_dir + '/' + env_name + '/'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        #### get number of log files in log directory
        current_num_files = next(os.walk(log_dir))[2]
        run_num = len(current_num_files)

        #### create new log file for each run
        log_f_name = log_dir + '/PPO_' + env_name + "_log_" + str(run_num) + ".csv"

        ################### checkpointing ###################
        run_num_pretrained = 0  #### change this to prevent overwriting weights in same env_name folder

        directory = "PPO_preTrained"
        if not os.path.exists(directory):
            os.makedirs(directory)

        directory = directory + '/' + env_name + '/'
        if not os.path.exists(directory):
            os.makedirs(directory)

        checkpoint_path = directory + "PPO_{}_{}.pth".format(env_name, run_num_pretrained)
        print("save checkpoint path : " + checkpoint_path)
        #####################################################

        # logging file
        log_f = open(log_f_name, "w+")
        log_f.write('episode,timestep,reward\n')

        # printing and logging variables
        print_running_reward = 0
        print_running_episodes = 0

        log_running_reward = 0
        log_running_episodes = 0

        time_step = 0

        start_time = datetime.now().replace(microsecond=0)

        update_timestep = 80
        epochs = 100
        # training loop
        for i in range(epochs):
            state = env.reset()
            current_ep_reward = 0
            while not done:
                try:
                    action = ppo_agent.select_action(state)
                    scaled_action = scale_actions_to_correct(np.squeeze(action.detach().numpy()))
                    state, reward, done, _, _ = env.step(scaled_action)

                    ppo_agent.buffer.rewards.append(reward)
                    ppo_agent.buffer.is_terminals.append(done)

                    time_step += 1
                    current_ep_reward += reward
                    # update PPO agent
                    if time_step % update_timestep == 0:
                        ppo_agent.update()

                    if time_step % action_std_decay_freq == 0:
                        ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

                    if time_step % log_freq == 0:
                        log_avg_reward = log_running_reward / log_running_episodes
                        log_avg_reward = round(log_avg_reward, 4)

                        log_f.write('{},{},{}\n'.format(i, time_step, log_avg_reward))
                        log_f.flush()

                        log_running_reward = 0
                        log_running_episodes = 0

                    if time_step % print_freq == 0:
                        print_avg_reward = print_running_reward / print_running_episodes
                        print_avg_reward = round(print_avg_reward, 2)

                        print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i, time_step,
                                                                                                print_avg_reward))

                        print_running_reward = 0
                        print_running_episodes = 0

                    if time_step % save_model_freq == 0:
                        print(
                            "--------------------------------------------------------------------------------------------")
                        print("saving model at : " + checkpoint_path)
                        ppo_agent.save(checkpoint_path)
                        print("model saved")
                        print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                        print(
                            "--------------------------------------------------------------------------------------------")

                    print_running_reward += current_ep_reward
                    print_running_episodes += 1

                    log_running_reward += current_ep_reward
                    log_running_episodes += 1


                except Exception as e:
                    print(e)
                    time.sleep(5.0)
                    break


if __name__ == "__main__":
    ex()
