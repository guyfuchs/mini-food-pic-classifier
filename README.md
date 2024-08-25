# mini-food-pic-classifier
This is a pictures classifier project based on PyTorch, implementing TinyVGG model from the CNN Explainer website. This limited classifier can handle 3 types of food pictures: pizza, steak and sushi.

This project is part of the course `Learn PyTorch for deep learning in a day. Literally.`

This repo was created as an educational project with 2 targets:
1. Have a basic model creation and training flow written in scripts.
2. Check a method to work with git from within Colab.

This repo contains 2 variations of the model construction:
* `notebooks/pytorch_going_modular_cell_mode.ipynb` - the model is built, trained and tested via the notebook.
* `notebooks/pytorch_going_modular_cell_mode.ipynb` - all model build, train and test code is written to python modules (scripts) in `scripts` folder and then called as a python script.

This second version enables running the model from command line as follows:
```
python scripts/train.py
```
**Note**: Data downloading was not turned modular yet, so the execution of the model training and testing can be done only after running the first chapter from the `notebooks/pytorch_going_modular_cell_mode.ipynb` notebook.

**Note**: Both notebooks contain also git related instructions, helped to update the git repo with the notebooks modifications and the generated modules. If you want to skip them, please remark them or remove them from the notebook.

**Note**: If you choose to work with the git in the notebook: 
1. Please make your own branch for the repo or your own clone in git
2. Update the repo name and/or branchs
3. Update the access credentials as detailed at the begining of each notebook.

##Preparations
Required preparations:
1. Have Python installed on the machine.
2. Required packages installed on the machine:
>>> pip install -r requirements.txt


