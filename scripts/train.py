import argparse
import diffuser.utils as utils
import pdb

#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

# Step 1: Pre-parse the command-line arguments to extract `exp_name`
pre_parser = argparse.ArgumentParser()
pre_parser.add_argument('--exp_name', required=True, help='Experiment name to specify the configuration.')
pre_args, remaining_args = pre_parser.parse_known_args()

# Step 2: Use the pre-parsed `exp_name` to define `config` dynamically
exp_name = pre_args.exp_name

class Parser(utils.Parser):
    dataset: str = 'maze2d-large-v1'
    config: str = f'config.{exp_name}'  # Dynamically incorporate `exp_name`

# Step 3: Parse the remaining arguments with the updated `Parser`
args = Parser().parse_args('diffusion')

#-----------------------------------------------------------------------------#
#---------------------------------- dataset ----------------------------------#
#-----------------------------------------------------------------------------#

dataset_config = utils.Config(
    args.loader,
    savepath=(args.savepath, 'dataset_config.pkl'),
    env=args.dataset,
    horizon=args.horizon,
    normalizer=args.normalizer,
    preprocess_fns=args.preprocess_fns,
    use_padding=args.use_padding,
    max_path_length=args.max_path_length,
)

render_config = utils.Config(
    args.renderer,
    savepath=(args.savepath, 'render_config.pkl'),
    env=args.dataset,
)

dataset = dataset_config()
renderer = render_config()

observation_dim = dataset.observation_dim
action_dim = dataset.action_dim

#-----------------------------------------------------------------------------#
#------------------------------ model & trainer ------------------------------#
#-----------------------------------------------------------------------------#

if exp_name=="maze2d" or exp_name=="maze2d_midgoal":
    model_config = utils.Config(
        args.model,
        savepath=(args.savepath, 'model_config.pkl'),
        horizon=args.horizon,
        transition_dim=observation_dim + action_dim,
        dim_mults=args.dim_mults,
        device=args.device,
    )
elif exp_name=="maze2d_condnet":
    model_config = utils.Config(
        args.model,
        savepath=(args.savepath, 'model_config.pkl'),
        horizon=args.horizon,
        transition_dim=observation_dim + action_dim,
        cond_dim = 4,
        dim_mults=args.dim_mults,
        device=args.device,
    )
elif exp_name=="maze2d_dual":
    model_config = utils.Config(
        args.model,
        savepath=(args.savepath, 'model_config.pkl'),
        horizon=args.horizon,
        transition_dim=2*(observation_dim + action_dim),
        dim_mults=args.dim_mults,
        device=args.device,
    )
elif exp_name=="maze2d_condnet_obstacle":
    model_config = utils.Config(
        args.model,
        savepath=(args.savepath, 'model_config.pkl'),
        horizon=args.horizon,
        transition_dim=observation_dim + action_dim,
        cond_dim=observation_dim,
        obstacle_dim=2,
        dim_mults=args.dim_mults,
        device=args.device,
    )
else:
    raise ValueError(f"Unrecognized experiment name: {exp_name}. Please check the configuration.")

diffusion_config = utils.Config(
    args.diffusion,
    savepath=(args.savepath, 'diffusion_config.pkl'),
    horizon=args.horizon,
    observation_dim=observation_dim,
    action_dim=action_dim,
    n_timesteps=args.n_diffusion_steps,
    loss_type=args.loss_type,
    clip_denoised=args.clip_denoised,
    predict_epsilon=args.predict_epsilon,
    ## loss weighting
    action_weight=args.action_weight,
    loss_weights=args.loss_weights,
    loss_discount=args.loss_discount,
    device=args.device,
)

trainer_config = utils.Config(
    utils.Trainer,
    savepath=(args.savepath, 'trainer_config.pkl'),
    train_batch_size=args.batch_size,
    train_lr=args.learning_rate,
    gradient_accumulate_every=args.gradient_accumulate_every,
    ema_decay=args.ema_decay,
    sample_freq=args.sample_freq,
    save_freq=args.save_freq,
    label_freq=int(args.n_train_steps // args.n_saves),
    save_parallel=args.save_parallel,
    results_folder=args.savepath,
    bucket=args.bucket,
    n_reference=args.n_reference,
    n_samples=args.n_samples,
    exp_name=exp_name,
)

#-----------------------------------------------------------------------------#
#-------------------------------- instantiate --------------------------------#
#-----------------------------------------------------------------------------#

model = model_config()

diffusion = diffusion_config(model)

trainer = trainer_config(diffusion, dataset, renderer)

#-----------------------------------------------------------------------------#
#------------------------ test forward & backward pass -----------------------#
#-----------------------------------------------------------------------------#

utils.report_parameters(model)

print('Testing forward...', end=' ', flush=True)
batch = utils.batchify(dataset[0])
loss, _ = diffusion.loss(*batch)
loss.backward()
print('✓')

#-----------------------------------------------------------------------------#
#--------------------------------- main loop ---------------------------------#
#-----------------------------------------------------------------------------#

n_epochs = int(args.n_train_steps // args.n_steps_per_epoch)

for i in range(n_epochs):
    print(f'Epoch {i} / {n_epochs} | {args.savepath}')
    trainer.train(n_train_steps=args.n_steps_per_epoch)

