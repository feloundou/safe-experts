# Main entrance of GAIL
import numpy as np
import gym
import safety_gym
import time, random, torch, wandb
from torch.distributions import Independent, OneHotCategorical, Categorical
import torch.nn.functional as F

import wandb.plot as wplot
from torch.optim import Adam

from neural_nets import VAELOR, ValorDiscriminator
from utils import mpi_fork, proc_id, num_procs, EpochLogger, \
    setup_pytorch_for_mpi, sync_params, mpi_avg_grads, count_vars, MemoryBatch


####################################################3

def vanilla_valor(env_fn,
                  vae=VAELOR,
                  disc = ValorDiscriminator,
                  dc_kwargs=dict(),
                  seed=0,
                  episodes_per_epoch=40,
                  epochs=50,
                  vae_lr=3e-4,
                  train_batch_size = 50,
                  eval_batch_size=200,
                  train_valor_iters=200,
                  max_ep_len=20,
                  logger_kwargs=dict(),
                  config_name='standard',
                  save_freq=10, replay_buffers=[], memories=[]):
    # W&B Logging
    wandb.login()

    composite_name = 'new_valor_' + config_name
    wandb.init(project="LearningCurves", group="Vanilla VALOR Expert", name=composite_name)

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

    # name = input("What is your name? ")
    # if (name != ""):
    #     # Print welcome message if the value is not empty
    #     print("Hello %s, welcome to playing with VAELOR" % name)

    # Model    # Create discriminator and monitor it
    con_dim = len(replay_buffers)
    # valor_vae = vae(obs_dim=env.observation_space.shape[0], latent_dim=con_dim, act_dim=2)
    valor_vae = vae(obs_dim=env.observation_space.shape[0], latent_dim=con_dim)

    con_dim = len(replay_buffers)
    vae_discrim = disc(input_dim=obs_dim[0], context_dim=con_dim, **dc_kwargs)

    # Set up model saving
    logger.setup_pytorch_saver([valor_vae])

    # Sync params across processes
    sync_params(valor_vae)

    N_expert = episodes_per_epoch*1000



    # Buffer
    local_episodes_per_epoch = int(episodes_per_epoch / num_procs())

    # Count variables
    var_counts = tuple(count_vars(module) for module in [valor_vae])
    logger.log('\nNumber of parameters: \t d: %d\n' % var_counts)

    # Optimizers
    vae_optimizer = Adam(valor_vae.parameters(), lr=vae_lr)
    discrim_optimizer = Adam(vae_discrim.parameters(), lr=vae_lr)

    start_time = time.time()

    # Prepare data
    mem = MemoryBatch(memories)
    transition_states, pure_states, transition_actions, expert_ids = mem.collate()

    valor_l_old, recon_l_old, context_l_old = 0, 0, 0

    # context_dist = OneHotCategorical(logits=torch.Tensor(np.ones(2)))
    context_dist = Categorical(logits=torch.Tensor(np.ones(2)))


# Main Loop
    for epoch in range(epochs):
        c = context_dist.sample()
        c_onehot = F.one_hot(c, con_dim).squeeze().float()

        o_tensor = context_dist.sample_n(train_batch_size)
        o_onehot = F.one_hot(o_tensor, con_dim).squeeze().float()

        # Select state transitions and actions at random indexes
        batch_indexes = torch.randint(len(transition_states), (train_batch_size,))
        raw_states_batch, delta_states_batch, actions_batch, sampled_experts = \
           pure_states[batch_indexes], transition_states[batch_indexes], transition_actions[batch_indexes], expert_ids[batch_indexes]

        # Train the VAE encoder and decoder
        for _ in range(train_valor_iters):
            valor_vae.train()
            vae_optimizer.zero_grad()
            loss, recon_loss, context_loss, _, _ = valor_vae.compute_latent_loss(raw_states_batch, delta_states_batch,
                                                                                 actions_batch, c_onehot, 'MSELoss')

            # loss, recon_loss, context_loss, _, _ = valor_vae.compute_latent_loss(raw_states_batch, delta_states_batch,
            #                                                                      actions_batch, o_onehot, 'MSELoss')

            loss.backward()
            vae_optimizer.step()
            valor_l = loss.data.item()


        valor_l_new, recon_l_new, context_l_new = valor_l, recon_loss.data.item(), context_loss.data.item()

        vaelor_metrics = {'VALOR Loss': valor_l, 'Recon Loss': recon_l_new, 'Context Loss': context_l_new}
        wandb.log(vaelor_metrics)

        logger.store(VALORLoss=valor_l_new, PolicyLoss=recon_l_new, ContextLoss=context_l_new,
                     DeltaValorLoss=valor_l_new-valor_l_old, DeltaPolicyLoss=recon_l_new-recon_l_old,
                     DeltaContextLoss=context_l_new-context_l_old
                     )

        # logger.store(VALORLoss = d_loss)
        valor_l_old, recon_l_old, context_l_old = valor_l_new, recon_l_new, context_l_new

        if (epoch % save_freq == 0) or (epoch == epochs - 1):
            logger.save_state({'env': env}, [valor_vae], None)

        # # Log
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpochBatchSize', train_batch_size)
        logger.log_tabular('VALORLoss', average_only=True)
        logger.log_tabular('PolicyLoss', average_only=True)
        logger.log_tabular('ContextLoss', average_only=True)
        # logger.log_tabular('DeltaValorLoss', average_only=True)
        # logger.log_tabular('DeltaPolicyLoss', average_only=True)
        # logger.log_tabular('DeltaContextLoss', average_only=True)
        logger.log_tabular('Time', time.time() - start_time)
        logger.dump_tabular()

    # Run eval
    print("RUNNING FINAL EVAL")
    print("Total episodes per expert: ", N_expert)
    valor_vae.eval()
    fake_c = context_dist.sample()

    print("fake c: ", fake_c)
    fake_c_onehot = F.one_hot(fake_c, con_dim).squeeze().float()
    print("fake c onehot: ", fake_c_onehot)

    # select a state transition, then see how it is labeled
    # batch_indexes = torch.randint(len(transition_states), (eval_batch_size,))
    eval_batch_index = None

    for i in range(len(memories)):
        curb_factor = episodes_per_epoch
        win_low = i*(N_expert-curb_factor)
        win_high = (i+1)*(N_expert-curb_factor)

        b_index = torch.randint(low=win_low, high=win_high, size=(eval_batch_size,))

        if eval_batch_index is None:
            eval_batch_index = b_index
        else:
            eval_batch_index = torch.cat([eval_batch_index, b_index])

    print("some batch index! ", eval_batch_index)

    # batch_index= torch.as_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10000, 10001, 10002, 10003, 10004, 10005, 10006, 10007, 10008, 10009])  # so first expert
    # print("old batching index! ", batch_index)
    eval_raw_states_batch, eval_delta_states_batch, eval_actions_batch, eval_sampled_experts = \
        pure_states[eval_batch_index], transition_states[eval_batch_index], transition_actions[eval_batch_index], expert_ids[
            eval_batch_index]

    # Pass through VAELOR
    loss, recon_loss, kl_loss, _, latent_v = valor_vae.compute_latent_loss(eval_raw_states_batch, eval_delta_states_batch,
                                                                                 eval_actions_batch, fake_c_onehot, 'MSELoss')

    print("Latent V: ", latent_v)
    predicted_expert_labels = np.argmax(latent_v, axis=1)  # convert from one-hot

    ground_truth, predictions = eval_sampled_experts, predicted_expert_labels

    print("ground truth", ground_truth)
    print("predictions ", predictions)

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
                  dc_kwargs=dict(hidden_dims=[args.hid] * args.l),
                  seed=args.seed, episodes_per_epoch=args.episodes_per_epoch,
                  epochs=args.epochs,
                  logger_kwargs=logger_kwargs)
