# Setup
Install the text-prompting repository. We recommend using pm2 to manage your processes. Install [here](https://pm2.io/docs/runtime/guide/installation/)

We will show how to run a vicuna miner below:

```bash
python -m pip install -e ~/text_prompting
```

Install the requirements for your specific miner. (if they exist)
```bash
python -m pip install -r ~/text_prompting/miners/vicuna/requirements.txt
```

Load it up:
```bash
CUDA_VISIBLE_DEVICES=0 pm2 start ~/text_prompting/neurons/miners/vicuna/miner.py \\
--name vicuna \\
--interpreter <path-to-python-binary> -- \\
--vicuna.model_name TheBloke/Wizard-Vicuna-7B-Uncensored-HF \\ # This can be changed to any Vicuna style model
--wallet.name <wallet-name> \\
--wallet.hotkey <wallet-hotkey> \\
--netuid <netuid> \\ # netuid 8 is on testnet currently
--subtensor.network <network> \\ (finney, test, local, etc)
--logging.debug \\ # set your desired logging level
--axon.port <port>
```