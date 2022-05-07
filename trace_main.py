import gym
import numpy as np
import itertools
import datetime
import torch
import torch.nn as nn
import copy
import time

from trace_sac import SAC
from torch.utils.tensorboard import SummaryWriter
from replay_buffer import ReplayMemory

from stock_trading_env import StockTradingEnv


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# TODO: 1. episode may end earlier when samples in the replayer buffer do not reach
#          the batch size, case1: ends at 420 episode step because of total asset < 0

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--hid', type=int, default=128)                          # hidden_size of fc
    parser.add_argument('--look_back', type=int, default=8)
    parser.add_argument('--num_stock', type=int, default=1)
    parser.add_argument('--balance',type=int, default=100000)

    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--seed', '-s', type=int, default=123)
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--updates_per_step', type=int, default=1)               # how many times to update in one step in env
    parser.add_argument('--target_update_interval', type=int, default = 1)       # how often to update the target value network
    parser.add_argument('--policy_delay', type=int, default = 1)                 # how often to update the policy network

    parser.add_argument('--policy_type', type=str, default='Gaussian')           # includes ["Gaussian", "Deterministic"]
    parser.add_argument('--trace_type', type=str, default='retrace')             # includes ['retrace','q_lambda', 'tree_backup','IS']
    parser.add_argument('--model_type', type=str, default='lstm')
    parser.add_argument('--automatic_entropy_tuning', type=str2bool, nargs='?',
                        const=True, default=True)

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--replay_size', type=int, default=100000)
    parser.add_argument('--cuda',type=str2bool,nargs='?',default=False)

    parser.add_argument('--lr_actor', type = float, default = 0.0005)
    parser.add_argument('--lr_critic', type = float, default = 0.0003)
    parser.add_argument('--lr_alpha', type = float, default = 0.003)
    parser.add_argument('--nsteps', type=int, default=30)
    parser.add_argument('--delay', type=int, default=1)
    parser.add_argument('--cbar', type=float, default=1.0)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--lambda_', type=float, default=0.8)


    parser.add_argument('--reps',type=int, default = 5)
    parser.add_argument('--env',type=int, default = 0)
    args = parser.parse_args()

    trace_list = ["retrace","qlambda", "treebackup","IS"]
    policy_list = ["Gaussian", "Deterministic"]
    assert args.policy_type in policy_list
    assert args.trace_type in trace_list

    # Settings
    SEEDS = args.seed
    replay_size = args.replay_size

    look_back = args.look_back       # size of extracted window
    updates_per_step = args.updates_per_step # model updates per simulator step (default: 1)
    num_stock = args.num_stock
    balance = args.balance

    hidden_size = args.hid

    gamma = args.gamma       # discount factor
    tau = args.tau       # target function update parameter
    lr_actor = args.lr_actor
    lr_critic = args.lr_critic
    lr_alpha = args.lr_alpha
    alpha = args.alpha        # weight of entropy term

    h_max = 1          # upper/lower bound of action
    batch_size = args.batch_size
    num_episodes = args.episodes
    target_update_interval = args.target_update_interval
    policy_delay = args.policy_delay

    reps = args.reps

    nsteps = args.nsteps
    lambda_ = args.lambda_
    cbar = args.cbar
    automatic_entropy_tuning = args.automatic_entropy_tuning


    policy_type = args.policy_type
    trace_type = args.trace_type
    model_type = args.model_type

    ## use gpu or not
    cuda = args.cuda

    env_num = args.env

    suffix = "SAC_{}_{}_{}_step{}_lam{}_{}_win{}_env{}".format(policy_type,
                "autotune" if automatic_entropy_tuning and policy_type == "Gaussian" else "",
                            trace_type if trace_type in trace_list else "",
                                        nsteps,
                lambda_ if trace_type == "retrace" or trace_type == "qlambda" else "",
                                        model_type,
                                        look_back,
                                        env_num)
    words = "======= We are training {} =======".format(suffix)
    print("="*len(words))
    print("="*len(words))
    print(words)
    print("="*len(words))
    print("="*len(words))

    # ============ Load data ================ #

    # PREFIX = '.'
    PREFIX = 'data2/week%d'%(env_num)
    train_trade_path = PREFIX+ '/train_trade_table.npy'
    val_trade_path = PREFIX+'/val_trade_table.npy'
    train_feature_path = PREFIX+'/train_feature_array_standard.npy'
    val_feature_path = PREFIX+'/val_feature_array_standard.npy'


    train_trade = np.load(train_trade_path)
    val_trade = np.load(val_trade_path)
    train_feature = np.load(train_feature_path)
    val_feature = np.load(val_feature_path)


    # concat the feature and the original information
    train_states = np.concatenate([train_feature,train_trade],axis=1)
    val_states = np.concatenate([val_feature,val_trade],axis=1)

    num_feature = train_states.shape[1]


    state_dim = num_stock*num_feature + 1 + num_stock # original states + balance + shares
    val_steps = len(val_trade) - look_back - 1


    # ============ Prepare Env ================ #
    env = StockTradingEnv(train_states,
                      look_back = look_back,
                      feature_num = num_feature,
                      steps = 2880-look_back - nsteps,
                      valid_env = True,
                      balance_init = balance)


    env_val = StockTradingEnv(val_states,
                              look_back = look_back,
                              steps = val_steps,
                              feature_num = num_feature,
                              valid_env = True,
                              balance_init = balance)
    # ============ Start training ================ #

    for i in range(reps):
        memory = ReplayMemory(replay_size,SEEDS)
        writer = SummaryWriter('runs/{}_{}_rep{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                                                suffix, i))
        agent = SAC(state_dim,
                    env.action_space,
                    look_back = look_back,
                    lr_actor = lr_actor,
                    lr_critic = lr_critic,
                    lr_alpha = lr_alpha,
                    target_update_interval = target_update_interval,
                    policy_delay = policy_delay,
                    cuda = cuda,
                    automatic_entropy_tuning=automatic_entropy_tuning,
                    tau = tau,
                    alpha = alpha,
                    gamma = gamma,
                    hidden_size=hidden_size,
                    nsteps=nsteps,
                    cbar=cbar,
                    lambda_= lambda_,
                    policy_type=policy_type,
                    model_type = model_type,
                    balance_init = balance,
                    num_episodes=num_episodes)


        total_numsteps = 0
        updates = 0


        for i_episode in itertools.count(1):
            episode_reward = 0
            episode_steps = 0
            done = False
            pre_time = time.time()

            state, _ = env.reset()
            while not done:

                obs_all = []
                action_all = []
                log_pi_all = []
                next_obs_all = []
                rews_all = []
                done_all = []

                for step in range(nsteps):
                    state_ = [state]
                    action,log_pi = agent.select_action(state_)  # Sample action from policy


                    # print("updates:",updates,"action: ",action[0], 'env steps', env.src.step)
                    next_state, reward, done, _ = env.step(action) # Step
                    if done:
                        break

                    episode_reward += reward
                    episode_steps += 1

                    obs_all.append(state)
                    action_all.append(action)
                    log_pi_all.append(log_pi)
                    rews_all.append(reward)
                    next_obs_all.append(next_state)
                    done_all.append(done)

                    state = next_state
                # important to make a deep copy of the state
                if len(obs_all) == nsteps:
                    memory.push(copy.deepcopy(obs_all), action_all, log_pi_all,
                                            rews_all, copy.deepcopy(next_obs_all), done_all) # Append transition to memory
                if len(memory) > batch_size:
                    # Number of updates per step in environment
                    for i in range(updates_per_step):
                        # Update parameters of all the networks
                        critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, batch_size, updates)

                        updates += 1
                        agent.critic_lr_scheduler.step()
                        agent.policy_lr_scheduler.step()



            # learning rate scheduling
        #     agent.critic_lr_scheduler.step()
        #     agent.policy_lr_scheduler.step()
            total_numsteps += 1
            market_gains = np.sum(np.vstack([info['market_gain'] for info in env.infos]),axis=0)[0]
            print("-------------------------------------------")
            print("Training Episode: {:d}, Avg Episode Reward: {:4f}(%%), Market-Gain: {:4f}(%%), elapase:{:4f}s".format(total_numsteps,
                                                                                        1e+4 * episode_reward /episode_steps,
                                                                                        1e+4 * market_gains / episode_steps,
                                                                                        time.time() - pre_time))
            print("Critic 1 Loss: {:.4e}, Critic 2 loss: {:.4e}, policy_loss: {:.4f}, ent loss: {:.4f}".format(critic_1_loss,
                                                                                                critic_2_loss,
                                                                                                policy_loss,
                                                                                                ent_loss))
            print("-------------------------------------------")
            writer.add_scalar('critic1_loss/train', critic_1_loss, total_numsteps)
            writer.add_scalar('critic2_loss/train', critic_2_loss, total_numsteps)
            writer.add_scalar('policy_loss/train', policy_loss, total_numsteps)
            writer.add_scalar('ent_loss/train', ent_loss, total_numsteps)
            if total_numsteps > num_episodes:
                break


            ## validating training every 10 episodoes
            if i_episode % 5 == 0:
                avg_reward = 0.
                episodes = 10
                for _  in range(episodes):
                    state, _ = env_val.reset()
                    state_ = [state]
                    episode_reward = 0
                    done = False
                    while not done:
                        action, _ = agent.select_action(state_, evaluate=True)
                        next_state, reward, done, _ = env_val.step(action)
                        episode_reward += reward


                        state = next_state
                        # print("action: ",action[0], 'env steps', env_val.src.step)
                    avg_reward += episode_reward
                avg_reward /= episodes

                writer.add_scalar('avg_reward/test', 1e+4*avg_reward, i_episode)
                market_gains = np.sum(np.vstack([info['market_gain'] for info in env_val.infos]),axis=0)[0]
                print("-------------------------------------------")
                print("Testing Episode: {:d}, Episode Reward(%%): {:4f},Market-Gain(%%): {:4f} ".format(i_episode,
                                                                                            1e+4 * avg_reward,
                                                                                             1e+4 * market_gains))
                print("-------------------------------------------")

        agent.save_checkpoint(suffix = suffix)
