**Setup UV and depandencies**

First, install `uv` following the [official installation guide](https://docs.astral.sh/uv/getting-started/installation/) (Windows, macOS & Linux supported)

Then this command will create your virtual environment with all the dependencies that are required

>uv sync

**Run program**

With VSCode, configure Python Runner as `.venv/bin/python3.12` (bottom right)

Without VSCode two choice :
1. Manual (after)
```
source .venv/bin/activate
python ./main.py
```
2. With uv
```
uv run main.py
```

**Load model**

This is for MLP models buts replace the 3 "MLP" by "GRU" and you got it for GRU

``` 
from MLP import MLP
import torch

my_loaded_model = MLP()
weight = torch.load("model_name.pth")
my_loaded_model.load_state_dict(weight)
my_loaded_model.eval()
```