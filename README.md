# RL Algorithm

It's a repository where i share reinforcement learning algorithms and some environments.  

## Development Environment

Create an anaconda environment using this command:

```
conda env create -f conda_env.yaml
```

If you want to change the environment name, change the value of `name` field, currently `rl-algorithm` value, in `conda_env.yaml` file.

## Structure

* [rl](/rl/) - Default RL module
  * [environment](/rl/environment/) - Environments for RL
  * [rl_algorithm](/rl/rl_algorithm/) - RL Algorithms
* [trainings](/trainings/) - Training files