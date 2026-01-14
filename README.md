**Setup UV and depandencies**

First, install `uv` following the [official installation guide](https://docs.astral.sh/uv/getting-started/installation/) (Windows, macOS & Linux supported)

Then this command will create your virtual environment with all the dependencies that are required

>uv sync

**Generating datasets**

This is done in an other repository : https://github.com/Gabiru1089/ComputerVisionProject

You can run the notebook from the second repository on Google Collab in order to obtain the two ".pkl" files

If you don't want to re-generate them, they are already in this repository

**Train model**

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

The four model that are trained are stored into ".pth" files

If you don't want to re-train them, they are already in this repository

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

**Generate analyse images**

>uv run plot_curves.py

Resulting images will be created inside example_pngs

If you don't want to regenerate them, they are already on the repo