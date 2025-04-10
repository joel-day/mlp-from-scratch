# MLP from Scrach using Numpy (HealthScore)

- HealthScore is a multi-layer perceptron trained using ONLY numpy which uses age, fitness, and diet to predict a health score from 1–10. 

- The training data was made up - by me - but created to mimic the relationships of each feature in the real world. 

- HealthScore is able to capture the effects of each feature (ie higher age = lower health score, higher diet = higher health score, higher fitness = higher health score).


## Installation

```bash
# Clone the repository
git clone https://github.com/joel-day/mlp-from-scratch.git

# Move into the local repository
cd mlp-from-scratch

# Create the virtual environment
uv venv .venv

# Activate the virtual environment
source .venv\bin\activate # Mac/Linux
.venv\Scripts\activate   # Windows

# Sync environment based on dependencies across all packages' pyproject.toml files
uv sync --all-packages
```

## Run the Model

- Run the model in the 'mlp.ipynb' notebook

- Execute the python code below in the 'mlp.ipynb' notebook to use the model

```python
health_score = mlp_health_score(age, health, diet)
```

## Understand How it Works

- The 'train_model.py' holds the code to train the network and provde the predictions.

- The 'mlp_scratch_numpy.ipynb' notebook is a more in depth workflow of the code. It has two parts 1) A simpler ANN model as the foundation 2) The MLP with added layers.
