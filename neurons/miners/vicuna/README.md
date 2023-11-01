## Vicuna Miner
Vicuna Language Model Serving with BitTensor
This code is for running the Vicuna model through the BitTensor framework.

# Overview

## Contents

- [Installing Dependencies](#installing-dependencies)
- [Converting Weights Into Model](#converting-weights-into-model)
- [Starting Miner](#starting-miner)


# Installing Dependencies

```
python3 -m pip install -r neurons/miners/vicuna/requirements.txt
```

# Converting Weights Into Model
If you already have a converted checkpoint of the model, you can skip this step.

## Vicuna Weights
The [Vicuna](https://vicuna.lmsys.org/) weights as delta weights to comply with the LLaMA model license.
You can add our delta to the original LLaMA weights to obtain the Vicuna weights. Instructions:

1. Get the original LLaMA weights in the huggingface format by following the instructions [here](https://huggingface.co/docs/transformers/main/model_doc/llama).
2. Use the following scripts to get Vicuna weights by applying our delta. They will automatically download delta weights from our Hugging Face [account](https://huggingface.co/lmsys).

**NOTE**:
Weights v1.1 are only compatible with the latest main branch of huggingface/transformers and ``fschat >= 0.2.0``.
Please update your local packages accordingly. If you follow the above commands to do a fresh install, then you should get all the correct versions.

Depending on which conversion script was used to create the Huggingface checkpoint of Llama, you might get an error that the tokenizer can not be found when loading the tokenizer. You can then replace all AutoTokenizers command with the correct tokenizer (in the example "LlamaTokenizer"), using this command:
```
find /path/to/fastchat -type f -name '*.py' -exec sed -i 's/AutoTokenizer/LlamaTokenizer/g' {} +
```

### Vicuna-7B
This conversion command needs around 30 GB of CPU RAM.
If you do not have enough memory, you can create a large swap file that allows the operating system to automatically utilize the disk as virtual memory.
```bash
python3 -m fastchat.model.apply_delta \
    --base /path/to/llama-7b \
    --target /output/path/to/vicuna-7b \
    --delta lmsys/vicuna-7b-delta-v1.1
```

### Vicuna-13B
This conversion command needs around 60 GB of CPU RAM.
If you do not have enough memory, you can create a large swap file that allows the operating system to automatically utilize the disk as virtual memory.
```bash
python3 -m fastchat.model.apply_delta \
    --base /path/to/llama-13b \
    --target /output/path/to/vicuna-13b \
    --delta lmsys/vicuna-13b-delta-v1.1
```


# Starting Miner
```
python3 neurons/miners/vicuna/miner.py
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