# Setup

We recommend using `pm2` to manage your processes. See the pm2 install [guide](https://pm2.io/docs/runtime/guide/installation/) for more info.

## Hardware requirements:
- Recommended: A100 80GB
- Minimum: A40 48GB or A6000 48GB

## Run command
`CUDA_VISIBLE_DEVICES=0 pm2 start ~/text_prompting/neurons/validators/validator.py \\ # path to the repo
    --name <your-validator-name> \\
    --interpreter <path-to-your-env-python> -- \\
    --wallet.name <validator-wallet> --netuid <netuid> \\ # netuid 8 is currently test netuid
    --wallet.hotkey <validator-hotkey> \\
    --subtensor.network <network> \\ (finney, local, test, etc)
    --logging.debug \\ # set desired logging level
    --neuron.reward_path ~/.bittensor/validators \\ # where to store logs
    --axon.port <port> \\
    --neuron.followup_sample_size <k> \\# This sets top-k for the followup prompts
    --neuron.answer_sample_size <k> \\ # This sets top-k for answer prompts


> Note: Make sure you have at least >50GB free disk space for wandb logs.