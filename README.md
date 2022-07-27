# RL Algorithm

It's a repository where i share reinforcement learning algorithms and some environments.  

## Development Environment

Create an anaconda environment using this command:

```
conda env create -f conda_env.yaml
```

If you want to change the environment name, change the value of `name` field, currently `rl-algorithm` value, in `conda_env.yaml` file.

If it doesn't work, then install manually using this commands:

```
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
python -m pip install mlagents==0.28.0
conda install ipykernel
conda install matplotlib
pip install tqdm
conda install -c conda-forge ipywidgets
pip install gym
pip install gym[classic_control]
```

## Structure

`*` marker indicates that you can use API from both that module and `rl` module.

* [rl](/rl/) - Default RL module
  * [*agent](/rl/agent/) - Agent Interfaces
  * [*drl_agent](/rl/drl_agent/) - DRL Agents
  * [environment](/rl/environment/) - Environments for RL
  * [*rl_agent](/rl/rl_agent/) - RL Agents
  * [*rl_util](/rl/rl_util/) - RL Utilities
  * [util](/rl/util/) - General Utilities
* [trainings](/trainings/) - Training files
