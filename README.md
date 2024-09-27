# KaMLs for Protein Ionization State Predictions: Are Trees All You Need?

Machine Learning models for pKa predictions of amino acid side-chains.

## KaML-Trees

All training and test data splits as well as pretrained KaML-CBtree models.

[CBTrees/train_test_splits](CBTrees/train_test_split) contains the train/test sets for each of the 20 splits

[CBTrees/models](CBTrees/models) contains the different tree models.

[CBTrees/calc_feat.py](CBTrees/calc_features.py) python script for feature calculation (have to polish it a bit more before uploading)

## KaML-GAT

[KaML-GAT/train_val_test](KaML-GAT/train_val_test) contains the datasets for training, validation and test for the 20 indenpendent splits. (exptAAfB_train/validation.csv, AA: AA_th split ID, B: B_th fold)

[KaML-GAT/model_inputs](KaML-GAT/model_inputs) contains the input files (constructured graph for each residue) for training, validation and test for the 20 indenpendent splits
