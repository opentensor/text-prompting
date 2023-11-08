# OpenAI Bittensor Miner
This repository contains a Bittensor Miner that uses OpenAI's GPT-3.5-turbo model as its synapse. The miner connects to the Bittensor network, registers its wallet, and serves the GPT-3.5-turbo model to the network by attaching the prompt function to the axon.

## Prerequisites

- Python 3.8+
- OpenAI Python API (https://github.com/openai/openai)

## Installation

1. Clone the repository
2. Install the required packages with `pip install -r requirements.txt`
3. Ensure that you have your OpenAI key in your os environment variable
```bash
# Sets your openai key in os envs variable
export OPENAI_API_KEY='your_openai_key_here'

# Verifies if openai key is set correctly
echo $OPENAI_API_KEY
```

For more configuration options related to the wallet, axon, subtensor, logging, and metagraph, please refer to the Bittensor documentation.

## Example Usage

To run the OpenAI Bittensor Miner with default settings, use the following command:

```
python3 -m pip install -r openminers/text_to_text/miner/openai/requirements.txt 
export OPENAI_API_KEY='sk-yourkey'
python3 neurons/miners/openai/miner.py
```

# Full Usage
```
usage: miner.py [-h] [--axon.port AXON.PORT] [--subtensor.network SUBTENSOR.NETWORK] [--subtensor.chain_endpoint SUBTENSOR.CHAIN_ENDPOINT] [--netuid NETUID] [--miner.root MINER.ROOT] [--miner.name MINER.NAME]
                [--miner.blocks_per_epoch MINER.BLOCKS_PER_EPOCH] [--miner.blacklist.blacklist [MINER.BLACKLIST.BLACKLIST ...]] [--miner.blacklist.whitelist [MINER.BLACKLIST.WHITELIST ...]]
                [--miner.blacklist.force_validator_permit] [--miner.blacklist.allow_non_registered] [--miner.blacklist.minimum_stake_requirement MINER.BLACKLIST.MINIMUM_STAKE_REQUIREMENT]
                [--miner.blacklist.prompt_cache_block_span MINER.BLACKLIST.PROMPT_CACHE_BLOCK_SPAN] [--miner.blacklist.use_prompt_cache] [--miner.blacklist.min_request_period MINER.BLACKLIST.MIN_REQUEST_PERIOD]
                [--miner.priority.default MINER.PRIORITY.DEFAULT] [--miner.priority.time_stake_multiplicate MINER.PRIORITY.TIME_STAKE_MULTIPLICATE]
                [--miner.priority.len_request_timestamps MINER.PRIORITY.LEN_REQUEST_TIMESTAMPS] [--miner.no_set_weights] [--miner.no_serve] [--miner.no_start_axon] [--miner.mock_subtensor] [--wandb.on]
                [--wandb.project_name WANDB.PROJECT_NAME] [--wandb.entity WANDB.ENTITY] [--logging.debug] [--logging.trace] [--logging.record_log] [--logging.logging_dir LOGGING.LOGGING_DIR] [--wallet.name WALLET.NAME]
                [--wallet.hotkey WALLET.HOTKEY] [--wallet.path WALLET.PATH] [--config CONFIG] [--strict] [--no_version_checking] [--no_prompt]

options:
  -h, --help            show this help message and exit
  --axon.port AXON.PORT
                        Port to run the axon on.
  --subtensor.network SUBTENSOR.NETWORK
                        Bittensor network to connect to.
  --subtensor.chain_endpoint SUBTENSOR.CHAIN_ENDPOINT
                        Chain endpoint to connect to.
  --netuid NETUID       The chain subnet uid.
  --miner.root MINER.ROOT
                        Trials for this miner go in miner.root / (wallet_cold - wallet_hot) / miner.name
  --miner.name MINER.NAME
                        Trials for this miner go in miner.root / (wallet_cold - wallet_hot) / miner.name
  --miner.blocks_per_epoch MINER.BLOCKS_PER_EPOCH
                        Blocks until the miner sets weights on chain
  --miner.blacklist.blacklist [MINER.BLACKLIST.BLACKLIST ...]
                        Blacklist certain hotkeys
  --miner.blacklist.whitelist [MINER.BLACKLIST.WHITELIST ...]
                        Whitelist certain hotkeys
  --miner.blacklist.force_validator_permit
                        Only allow requests from validators
  --miner.blacklist.allow_non_registered
                        If True, the miner will allow non-registered hotkeys to mine.
  --miner.blacklist.minimum_stake_requirement MINER.BLACKLIST.MINIMUM_STAKE_REQUIREMENT
                        Minimum stake requirement
  --miner.blacklist.prompt_cache_block_span MINER.BLACKLIST.PROMPT_CACHE_BLOCK_SPAN
                        Amount of blocks to keep a prompt in cache
  --miner.blacklist.use_prompt_cache
                        If True, the miner will use the prompt cache to store recent request prompts.
  --miner.blacklist.min_request_period MINER.BLACKLIST.MIN_REQUEST_PERIOD
                        Time period (in minute) to serve a maximum of 50 requests for each hotkey
  --miner.priority.default MINER.PRIORITY.DEFAULT
                        Default priority of non-registered requests
  --miner.priority.time_stake_multiplicate MINER.PRIORITY.TIME_STAKE_MULTIPLICATE
                        Time (in minute) it takes to make the stake twice more important in the priority queue
  --miner.priority.len_request_timestamps MINER.PRIORITY.LEN_REQUEST_TIMESTAMPS
                        Number of historic request timestamps to record
  --miner.no_set_weights
                        If True, the miner does not set weights.
  --miner.no_serve      If True, the miner doesnt serve the axon.
  --miner.no_start_axon
                        If True, the miner doesnt start the axon.
  --miner.mock_subtensor
                        If True, the miner will allow non-registered hotkeys to mine.
  --wandb.on            Turn on wandb.
  --wandb.project_name WANDB.PROJECT_NAME
                        The name of the project where youre sending the new run.
  --wandb.entity WANDB.ENTITY
                        An entity is a username or team name where youre sending runs.
  --logging.debug       Turn on bittensor debugging information
  --logging.trace       Turn on bittensor trace level information
  --logging.record_log  Turns on logging to file.
  --logging.logging_dir LOGGING.LOGGING_DIR
                        Logging default root directory.
  --wallet.name WALLET.NAME
                        The name of the wallet to unlock for running bittensor (name mock is reserved for mocking this wallet)
  --wallet.hotkey WALLET.HOTKEY
                        The name of the wallet's hotkey.
  --wallet.path WALLET.PATH
                        The path to your bittensor wallets
  --config CONFIG       If set, defaults are overridden by passed file.
  --strict              If flagged, config will check that only exact arguments have been set.
  --no_version_checking
                        Set true to stop cli version checking.
  --no_prompt           Set true to stop cli from prompting the user.
```