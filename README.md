# RL Algorithm

It's a repository where i share reinforcement learning algorithms and some environments.  

## Development Environment

Create an anaconda environment using this command:

```
conda env create -f conda_env.yaml
```

If you want to change the environment name, change the value of `name` field, currently `rl-algorithm` value, in `conda_env.yaml` file.

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
