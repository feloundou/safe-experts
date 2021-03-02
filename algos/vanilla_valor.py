# Main entrance of GAIL
import numpy as np
import gym
import safety_gym
import time, random, torch, wandb

import wandb.plot as wplot
import os.path as osp
from torch import nn
from torch.optim import Adam
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from neural_nets import ActorCritic, ValorDiscriminator,  VAELOR


from utils import PureVALORBuffer, mpi_fork, proc_id, num_procs, EpochLogger, \
    setup_pytorch_for_mpi, sync_params, mpi_avg_grads, count_vars


####################################################3

def vanilla_valor(env_fn,
                  # disc=ValorDiscriminator,
                  # con_labeler=MLPContextLabeler,
                  # vae = VanillaVAE,
                  vae=VAELOR,
                  # reward_labeler=GaussianReward,
                  # actor_critic=ValorActorCritic,
                  ac_kwargs=dict(), dc_kwargs=dict(),
                  seed=0,
                  episodes_per_epoch=40,
                  epochs=50,
                  dc_lr=5e-4, pi_lr=3e-4,
                  train_batch_size = 50,
                  eval_batch_size=200,
                  #
                  # train_dc_iters=10,
                  # train_dc_interv=1,
                  max_ep_len=20,
                  logger_kwargs=dict(),
                  config_name='standard',
                  # splitN=8,
                  save_freq=10, replay_buffers=[], memories=[]):
    # W&B Logging
    wandb.login()

    composite_name = 'new_valor_' + config_name
    wandb.init(project="LearningCurves", group="Steer VALOR Expert", name=composite_name)

    assert replay_buffers != [], "Replay buffers must be set"

    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    setup_pytorch_for_mpi()

    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Instantiate environment
    env = env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    # Model    # Create discriminator and monitor it
    con_dim = len(replay_buffers)
    valor_vae = vae(obs_dim=env.observation_space.shape[0], latent_dim=con_dim)

    # Set up model saving
    logger.setup_pytorch_saver([valor_vae])

    # Sync params across processes
    sync_params(valor_vae)

    # Buffer
    local_episodes_per_epoch = int(episodes_per_epoch / num_procs())
    # buffer = PureVALORBuffer(con_dim, obs_dim[0], act_dim[0], local_episodes_per_epoch, max_ep_len, train_dc_interv,
    #                          N=splitN)

    # Count variables
    var_counts = tuple(count_vars(module) for module in [valor_vae])
    logger.log('\nNumber of parameters: \t d: %d\n' % var_counts)

    # Optimizers
    vae_optimizer = Adam(valor_vae.parameters(), lr=pi_lr)

    start_time = time.time()
    # context_dist = Categorical(logits=torch.Tensor(np.ones(con_dim)))
    total_t, ep_len, total_r = 0, 0, 0

    # Prepare data
    expert_states, expert_actions, expert_state_diff = None, None, None
    t_states, t_actions, transition_states, transition_actions = None, None, None, None
    e_idx = 0

    for k in range(len(replay_buffers)):
        if transition_states is None:
            expert_states, expert_actions, _, _, expert_next_states, _ = memories[k].sample(next=True)
            # Exclude the last step of each episode to calculate state differences
            t_states = torch.stack(
                [expert_next_states[i] - expert_states[i] for episode in memories[k] for i in range(len(episode) - 1)])
            t_actions = torch.stack([expert_actions[i] for episode in memories[k] for i in range(len(episode) - 1)])

            # Three basic checks
            assert t_states.shape[0] == t_actions.shape[
                0], "Tensors for state transitions and actions should be same on dim 0"
            assert torch.equal(expert_next_states[0],
                               expert_states[1]), "The i+1 state tensors should match the i next_state tensors"
            assert torch.equal(expert_states[1] - expert_states[0], t_states[0]), "Check your transition calculations"

            transition_states, transition_actions = t_states, t_actions
            # expert_ids = torch.full_like(transition_states[0], e_idx)
            expert_ids = torch.empty(transition_states.shape[0]).fill_(e_idx)
            print("expert ids: ", expert_ids)
            print("expert size: ", expert_ids.shape)

        else:
            expert_states, expert_actions, _, _, expert_next_states, _ = memories[k].sample(next=True)
            # expert_states = torch.cat([expert_states, torch.Tensor(states)])
            # expert_actions = torch.cat([expert_actions, torch.Tensor(actions)])

            t_states = torch.stack([expert_next_states[i] - expert_states[i] for episode in memories[k] for i in
                                    range(len(episode) - 1)])
            t_actions = torch.stack([expert_actions[i] for episode in memories[k] for i in range(len(episode) - 1)])

            # Three basic checks
            assert t_states.shape[0] == t_actions.shape[
                0], "Tensors for state transitions and actions should be same on dim 0"
            assert torch.equal(expert_next_states[0],
                               expert_states[1]), "The i+1 state tensors should match the i next_state tensors"
            assert torch.equal(expert_states[1] - expert_states[0],
                               t_states[0]), "Check your transition calculations"

            transition_states = torch.cat([transition_states, t_states])
            transition_actions = torch.cat([transition_actions, t_actions])

            # e_ids = torch.full_like(transition_states[0], e_idx)

            e_ids = torch.empty(transition_states.shape[0]).fill_(e_idx)
            expert_ids = torch.cat([expert_ids, e_ids])
            print("expert ids: ", expert_ids)
            print("expert size: ", expert_ids.shape)

        e_idx += 1

    valor_l_old, recon_l_old, context_l_old = 0,0,0
    for epoch in range(epochs):
        print("local episodes:", local_episodes_per_epoch)

        N = expert_states.shape[0]
        print("N: ", N)
        # print("Final Transition state shapes: ", transition_states.shape)
        # print("Final Transition action shapes: ", transition_actions.shape)

        # Shuffle the transition (state-action) data to draw our sample
        # rand_index = torch.randperm(transition_states.size()[0])

        batch_indexes = torch.randint(len(transition_states), (train_batch_size,))
        # print("RANDOM INDEXES: ", rand_index)
        # print("BATCH INDEXES: ", batch_indexes)

        states_batch = transition_states[batch_indexes]
        actions_batch = transition_actions[batch_indexes]
        sampled_experts = expert_ids[batch_indexes]
        # print("Sampled expert ids : ", sampled_experts)

        # Train the VAE encoder and decoder
        train_valor_iters = 200

        for _ in range(train_valor_iters):
            vae_optimizer.zero_grad()
            loss, recon_loss, context_loss, _, _ = valor_vae.compute_latent_loss(states_batch, actions_batch)
            loss.backward()
            vae_optimizer.step()
            valor_l = loss.data.item()
            ep_len += 1



        valor_l_new , recon_l_new, context_l_new = valor_l, recon_loss.data.item(), context_loss.data.item()

        vaelor_metrics = {'VALOR Loss': valor_l, 'Recon Loss': recon_l_new, 'Context Loss': context_l_new}
        wandb.log(vaelor_metrics)

        logger.store(VALORLoss=valor_l_new, PolicyLoss=recon_l_new, ContextLoss=context_l_new,
                     DeltaValorLoss=valor_l_new-valor_l_old, DeltaPolicyLoss=recon_l_new-recon_l_old,
                     DeltaContextLoss=context_l_new-context_l_old
                     )
        valor_l_old, recon_l_old, context_l_old = valor_l_new, recon_l_new, context_l_new

        if (epoch % save_freq == 0) or (epoch == epochs - 1):
            logger.save_state({'env': env}, [valor_vae], None)

        # Log
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpochBatchSize', train_batch_size)
        logger.log_tabular('VALORLoss', average_only=True)
        logger.log_tabular('PolicyLoss', average_only=True)
        logger.log_tabular('ContextLoss', average_only=True)
        logger.log_tabular('DeltaValorLoss', average_only=True)
        logger.log_tabular('DeltaPolicyLoss', average_only=True)
        logger.log_tabular('DeltaContextLoss', average_only=True)
        logger.log_tabular('Time', time.time() - start_time)
        logger.dump_tabular()



    # Run eval
    print("RUNNING FINAL EVAL")

    batch_indexes = torch.randint(len(transition_states), (eval_batch_size,))

    states_eval_batch = transition_states[batch_indexes]
    actions_eval_batch = transition_actions[batch_indexes]
    sampled_eval_experts = expert_ids[batch_indexes]

    loss, recon_loss, kl_loss, _, latent_v = valor_vae.compute_latent_loss(states_eval_batch, actions_eval_batch)

    predicted_expert_labels = np.argmax(latent_v, axis=1)
    print("relabeled: ", predicted_expert_labels)

    ground_truth = sampled_eval_experts
    predictions = predicted_expert_labels
    ep_len = 0

    # Confusion matrix
    class_names = ["0", "1"]
    wandb.log({"confusion_mat": wplot.confusion_matrix(
        y_true=np.array(ground_truth), preds=np.array(predictions), class_names=class_names)})

    wandb.finish()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Safexp-PointGoal1-v0')
    parser.add_argument('--hid', type=int, default=128)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--episodes-per-epoch', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--exp_name', type=str, default='valor-anonymous-expert')
    args = parser.parse_args()

    mpi_fork(args.cpu)

    from utils import setup_logger_kwargs

    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    vanilla_valor(lambda: gym.make(args.env),
                  # disc=ValorDiscriminator,
                  ac_kwargs=dict(hidden_sizes=[args.hid] * args.l),
                  dc_kwargs=dict(hidden_dims=[args.hid] * args.l),
                  seed=args.seed, episodes_per_epoch=args.episodes_per_epoch,
                  epochs=args.epochs,
                  logger_kwargs=logger_kwargs)
