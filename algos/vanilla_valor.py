# Main entrance of GAIL
import numpy as np
import gym
import safety_gym
import time, random, torch, wandb
from torch.distributions import Independent, OneHotCategorical, Categorical
import torch.nn.functional as F
import wandb.plot as wplot
from torch.optim import Adam, SGD, lr_scheduler
import numpy as np
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

from neural_nets import VAELOR, ValorDiscriminator, VQCriterion

from utils import mpi_fork, proc_id, num_procs, EpochLogger, \
    setup_pytorch_for_mpi, sync_params, mpi_avg_grads, count_vars, MemoryBatch, \
    frange_cycle_linear, frange_cycle_sigmoid


####################################################3

def vanilla_valor(env_fn,
                  vae=VAELOR,
                  vaelor_kwargs=dict(),
                  annealing_kwargs=dict(),
                  seed=0,
                  episodes_per_epoch=40,
                  epochs=50,
                  train_valor_iters=5,
                  vae_lr=1e-3,
                  train_batch_size=50,
                  eval_batch_size=200,
                  # train_valor_iters=200,
                  max_ep_len=1000,
                  logger_kwargs=dict(),
                  config_name='standard',
                  save_freq=10, replay_buffers=[], memories=[]):
    # W&B Logging
    wandb.login()

    composite_name = str(epochs) + 'epochs_' + str(train_batch_size) + 'batch_' + \
                     str(vaelor_kwargs['encoder_hidden']) + 'enc_' + str(vaelor_kwargs['decoder_hidden']) + 'dec'

    wandb.init(project="VQ VAELOR", group='Epochs: ' + str(epochs),
               name=composite_name, config=locals())

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
    # con_dim = 10   # con_dim = 1
    valor_vae = vae(obs_dim=obs_dim[0], latent_dim=con_dim, out_dim=act_dim[0], **vaelor_kwargs)

    # Set up model saving
    logger.setup_pytorch_saver([valor_vae])

    # Sync params across processes
    sync_params(valor_vae)
    N_expert = episodes_per_epoch*max_ep_len

    # Buffer
    local_episodes_per_epoch = int(episodes_per_epoch / num_procs())

    # Count variables
    var_counts = tuple(count_vars(module) for module in [valor_vae])
    logger.log('\nNumber of parameters: \t d: %d\n' % var_counts)

    # Optimizers
    vae_optimizer = Adam(valor_vae.parameters(), lr=vae_lr)

    start_time = time.time()

    # Prepare data
    mem = MemoryBatch(memories)

    transition_states, pure_states, transition_actions, expert_ids = mem.collate()
    valor_l_old, recon_l_old, context_l_old = 0, 0, 0

    # context_dist = OneHotCategorical(logits=torch.Tensor(np.ones(2)))
    context_dist = Categorical(logits=torch.Tensor(np.ones(con_dim)))

    # Main Loop
    kl_beta_schedule = frange_cycle_sigmoid(epochs, **annealing_kwargs)   # kl_beta_schedule = frange_cycle_linear(epochs, n_cycle=10)
    for epoch in range(epochs):
        valor_vae.train()
        ##
        # c = context_dist.sample()  # this causes context learning to collapse very quickly
        # c_onehot = F.one_hot(c, con_dim).squeeze().float()

        o_tensor = context_dist.sample((train_batch_size,))
        o_onehot = F.one_hot(o_tensor, con_dim).squeeze().float()

        # Select state transitions and actions at random indexes
        batch_indexes = torch.randint(len(transition_states), (train_batch_size,))

        raw_states_batch, delta_states_batch, actions_batch, sampled_experts = \
           pure_states[batch_indexes], transition_states[batch_indexes], transition_actions[batch_indexes], expert_ids[batch_indexes]

        print("Expert IDs: ", sampled_experts)

        for i in range(train_valor_iters):  # trying out iterations again
        # for i in range(20):  # trying out iterations again
            if i > 0:
                kl_beta = 0   # freeze context loss
            else:
                kl_beta = kl_beta_schedule[epoch]
            loss, recon_loss, _, _ = valor_vae.compute_latent_loss(raw_states_batch, delta_states_batch,
                                                                                 actions_batch, o_onehot, con_dim,
                                                                                 kl_beta)

            vae_optimizer.zero_grad()
            loss.mean().backward()
            vae_optimizer.step()

        valor_l_new, recon_l_new = loss.mean().data.item(), recon_loss.mean().data.item()
        # valor_l_new, recon_l_new, context_l_new = total_loss, recon_loss, vq_loss

        vaelor_metrics = {'VALOR Loss': valor_l_new, 'Recon Loss': recon_l_new, "KL Beta": kl_beta_schedule[epoch]}
        wandb.log(vaelor_metrics)

        logger.store(VALORLoss=valor_l_new, PolicyLoss=recon_l_new,
                     # ContextLoss=context_l_new,
                     DeltaValorLoss=valor_l_new-valor_l_old, DeltaPolicyLoss=recon_l_new-recon_l_old,
                     # DeltaContextLoss=context_l_new-context_l_old
                     )

        # logger.store(VALORLoss = d_loss)
        valor_l_old, recon_l_old = valor_l_new, recon_l_new
            # , context_l_new

        if (epoch % save_freq == 0) or (epoch == epochs - 1):
            logger.save_state({'env': env}, [valor_vae], None)

        # Log
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpochBatchSize', train_batch_size)
        logger.log_tabular('VALORLoss', average_only=True)
        logger.log_tabular('PolicyLoss', average_only=True)
        # logger.log_tabular('ContextLoss', average_only=True)
        # logger.log_tabular('DeltaValorLoss', average_only=True)
        # logger.log_tabular('DeltaPolicyLoss', average_only=True)
        # logger.log_tabular('DeltaContextLoss', average_only=True)
        logger.log_tabular('Time', time.time() - start_time)
        logger.dump_tabular()



#########
    # Run eval
    print("RUNNING Classification EVAL")
    valor_vae.eval()

    fake_o = context_dist.sample((eval_batch_size*len(replay_buffers),))  #eval batch size for each expert type
    fake_o_onehot = F.one_hot(fake_o, con_dim).squeeze().float()

    # Randomize and fetch an evaluation sample
    eval_raw_states_batch, eval_delta_states_batch, eval_actions_batch, eval_sampled_experts = \
         mem.eval_batch(N_expert, eval_batch_size, episodes_per_epoch)

    # Pass through VAELOR
    loss, recon_loss, _, predicted_expert_labels = valor_vae.compute_latent_loss(eval_raw_states_batch, eval_delta_states_batch,
                                                                                 eval_actions_batch, fake_o_onehot, con_dim)

    ground_truth, predictions = eval_sampled_experts, predicted_expert_labels

    print("ground truth", np.array(ground_truth))
    print("predictions ", np.array(predictions))

    # Confusion matrix
    class_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    wandb.log({"confusion_mat": wplot.confusion_matrix(
        y_true=np.array(ground_truth), preds=np.array(predictions), class_names=class_names)})

    print("RUNNING POLICY EVAL")  # unroll and plot a full episode # (for now, selecting first episode in first memory)
    # pick arbitary episode as starting point

    eval_observations, eval_actions, _, _ = memories[0].sample()    # 0th expert. Look at 0 or 1. TODO: Use both types of memories here.
    one_ep_states, one_ep_actions = eval_observations[:max_ep_len], eval_actions[:max_ep_len]  # first episode
    expert_x_actions, expert_y_actions = map(list, zip(*one_ep_actions))

    ep_time = torch.arange(max_ep_len)


    expert_x_table, expert_y_table = wandb.Table(data=[[x, y] for (x, y) in zip(ep_time, expert_x_actions)], columns=["x", "y"]), \
                       wandb.Table(data=[[x, y] for (x, y) in zip(ep_time, expert_y_actions)], columns=["x", "y"])

    # Collect learner experiences, give the network a state observation and a fixed label tensor
    expert_idx_list = np.arange(10)

    LearnerActions = [[] for i in range(10)]
    ExpertTypeTensors = [torch.reshape(torch.as_tensor(i), (-1,)) for i in expert_idx_list]
    Learner_X_Actions, Learner_Y_Actions, Learner_Data_X_Time, Learner_Data_Y_Time, Learner_X_Table, Learner_Y_Table = \
        [], [], [], [], [], []

    for step in range(max_ep_len):
        for k in range(10):
            ActionDist = valor_vae.decoder(torch.cat([one_ep_states[step], ExpertTypeTensors[k]], dim=-1))
            LearnerActions[k].append(ActionDist.sample())

    for k in range(10):
        X_Dim_Actions, Y_Dim_Actions = map(list, zip(*LearnerActions[k]))
        Learner_X_Actions.append(X_Dim_Actions)
        Learner_Y_Actions.append(Y_Dim_Actions)
        # Learner_Data_X_Time.append([[x, y] for (x, y) in zip(ep_time, X)])
        # Learner_Data_Y_Time.append([[x, y] for (x, y) in zip(ep_time, Y)])
        # Learner_X_Table.append(wandb.Table(data=Learner_Data_X_Time[k], columns=["x", "y"]))
        # Learner_Y_Table.append(wandb.Table(data=Learner_Data_Y_Time[k], columns=["x", "y"]))
        Learner_X_Table.append(wandb.Table(data=[[x, y] for (x, y) in zip(ep_time, X_Dim_Actions)], columns=["x", "y"]))
        Learner_Y_Table.append(wandb.Table(data=[[x, y] for (x, y) in zip(ep_time, Y_Dim_Actions)], columns=["x", "y"]))


    wandb.log({"Tracing Dimension 1": wandb.plot.scatter(expert_x_table, "x", "y",  title="Expert (X plane)"),
               "Tracing Dimension 2": wandb.plot.scatter(expert_y_table, "x", "y",  title="Expert (Y plane)"),
               "Learner0 Tracing Dimension 1": wandb.plot.scatter(Learner_X_Table[0], "x", "y", title="Student [0] (X plane)"),
               "Learner0 Tracing Dimension 2": wandb.plot.scatter(Learner_Y_Table[0], "x", "y", title="Student [0] (Y plane)"),
               "Learner1 Tracing Dimension 1": wandb.plot.scatter(Learner_X_Table[1], "x", "y", title="Student [1] (X plane)"),
               "Learner1 Tracing Dimension 2": wandb.plot.scatter(Learner_Y_Table[1], "x", "y", title="Student [1] (Y plane)"),
               "Learner2 Tracing Dimension 1": wandb.plot.scatter(Learner_X_Table[2], "x", "y", title="Student [2] (X plane)"),
               "Learner2 Tracing Dimension 2": wandb.plot.scatter(Learner_Y_Table[2], "x", "y", title="Student [2] (Y plane)"),
               "Learner3 Tracing Dimension 1": wandb.plot.scatter(Learner_X_Table[3], "x", "y", title="Student [3] (X plane)"),
               "Learner3 Tracing Dimension 2": wandb.plot.scatter(Learner_Y_Table[3], "x", "y", title="Student [3] (Y plane)"),
               "Learner4 Tracing Dimension 1": wandb.plot.scatter(Learner_X_Table[4], "x", "y", title="Student [4] (X plane)"),
               "Learner4 Tracing Dimension 2": wandb.plot.scatter(Learner_Y_Table[4], "x", "y", title="Student [4] (Y plane)"),
               "Learner5 Tracing Dimension 1": wandb.plot.scatter(Learner_X_Table[5], "x", "y", title="Student [5] (X plane)"),
               "Learner5 Tracing Dimension 2": wandb.plot.scatter(Learner_Y_Table[5], "x", "y", title="Student [5] (Y plane)"),
               "Learner6 Tracing Dimension 1": wandb.plot.scatter(Learner_X_Table[6], "x", "y", title="Student [6] (X plane)"),
               "Learner6 Tracing Dimension 2": wandb.plot.scatter(Learner_Y_Table[6], "x", "y", title="Student [6] (Y plane)"),
               "Learner7 Tracing Dimension 1": wandb.plot.scatter(Learner_X_Table[7], "x", "y", title="Student [7] (X plane)"),
               "Learner7 Tracing Dimension 2": wandb.plot.scatter(Learner_Y_Table[7], "x", "y", title="Student [7] (Y plane)"),
               "Learner8 Tracing Dimension 1": wandb.plot.scatter(Learner_X_Table[8], "x", "y", title="Student [8] (X plane)"),
               "Learner8 Tracing Dimension 2": wandb.plot.scatter(Learner_Y_Table[8], "x", "y", title="Student [8] (Y plane)"),
               "Learner9 Tracing Dimension 1": wandb.plot.scatter(Learner_X_Table[9], "x", "y", title="Student [9] (X plane)"),
               "Learner9 Tracing Dimension 2": wandb.plot.scatter(Learner_Y_Table[9], "x", "y", title="Student [9] (Y plane)"),
               })

    wandb.log({"Type 0 Learner": wandb.Video("/home/tyna/Documents/safe-experts/algos/recording/Type0_Learner.mp4", fps=4, format="gif")})
    wandb.log({"Type 5 Learner": wandb.Video("/home/tyna/Documents/safe-experts/algos/recording/Type5_Learner.mp4", fps=4, format="gif")})
    wandb.log({"Type 6 Learner": wandb.Video("/home/tyna/Documents/safe-experts/algos/recording/Type6_Learner.mp4", fps=4, format="gif")})

    # # Log points and boxes
    # # Create a figure and a 3D Axes
    # fig0 = plt.figure()
    # ax = Axes3D(fig0)
    #
    # # Create an init function and the animate functions. Since we are changing the elevation and azimuth and
    # # no objects are really changed on the plot we don't have to return anything from the init and animate function.
    # def init_expert():
    #     ax.scatter(expert_x_actions, ep_time, expert_y_actions, c=ep_time, cmap='inferno', linewidth=0.5, alpha=0.6)
    #     return figE,
    #
    # def image_init(index=0):
    #     ax.scatter(Learner_X_Actions[index], ep_time, Learner_Y_Actions[index], c=ep_time, cmap='viridis', linewidth=0.5,
    #                alpha=0.6)
    #     return fig0,
    #
    # def animate_exp(i):
    #     ax.view_init(elev=10., azim=i)
    #     return figE,
    #
    # def animate(i):
    #     ax.view_init(elev=10., azim=i)
    #     return fig0,
    #
    # # Animate
    # anim0 = animation.FuncAnimation(fig0, animate, init_func=lambda: image_init(0), frames=360, interval=20, blit=True)
    #
    # figE = plt.figure()
    # ax = Axes3D(figE)
    # anim_expert = animation.FuncAnimation(figE, animate_exp, init_func=init_expert, frames=360, interval=20, blit=True)
    #
    # # Save
    # root_dir = "/home/tyna/Documents/safe-experts/algos/"
    # f0 = root_dir + "student0_animation.mp4"
    # f_expert = root_dir + "expert_animation.mp4"
    #
    # writervideo = animation.FFMpegWriter(fps=60)
    # anim0.save(f0, writer=writervideo)
    # anim_expert.save(f_expert, writer=writervideo)
    #
    # wandb.log({"Expert":    wandb.Video(f_expert),  "Learner 0": wandb.Video(f0),
    #            })

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
                  seed=args.seed, episodes_per_epoch=args.episodes_per_epoch,
                  epochs=args.epochs,
                  logger_kwargs=logger_kwargs)
