# Geometric Deep Learning for Rubik's Cube Group
A repository for the project on using symmetry-aware deep learning architectures to solve the Rubik's cube.

To reproduce the results, download data from [Google Drive](https://drive.google.com/drive/folders/1jEhSZc4QXKETcHl7HUjaPRaa3KTS_5SI?usp=share_link) and save it to a `data` folder.
The data for the sampled dataset collected by [Herbert Kociemba](http://kociemba.org/) are available at [http://kociemba.org/math/optman/100000optcubes.zip](http://kociemba.org/math/optman/100000optcubes.zip)

The generated plots can be found at [Google Drive](https://drive.google.com/drive/u/1/folders/14ezlOzEoX2d5CWgkDuFBUcjBHScYstKC).

## Installation
Create a Python environment, e.g., with `venv`, and activate it: 
```
python -m venv venv
source venv/bin/activate
```
Install required libraries: 
```
pip install -r requirements.txt
```

## Project Structure

The project is structured the following way.
- cube data structures are implemented in folder `classes/`
- data generation and symmetry-equivalence compression is implemented in folder `generate/`, Python script `utils/compressions.py`, and Jupyter notebooks `generate.ipynb` and `weisfeiler-lehman_compressions.ipynb`
- PyTorch models and training runners are defined in `pytorch_classes/`
- all scripts run on the cluster are in folder `scripts/`
- analysis of the training and search results can be found in `analyzeResults.ipynb`, `summarize_accuracies.ipynb`, and `summarize_search.ipynb` Jupyter notebooks
- explainability experiments can be found in Jupyter notebook `explainer.ipynb`