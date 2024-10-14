# KaMLs for Protein Ionization State Predictions: Are Trees All You Need?

Machine Learning models for pKa predictions of amino acid side-chains.

## KaML-Trees

All training and test data splits as well as pretrained KaML-CBtree models.

[CBTrees/train_test_splits](CBTrees/train_test_split) contains the train/test sets for each of the 20 splits

[CBTrees/models](CBTrees/models) contains the CBTree models for each of the 20 test splits as well ass the finalized models  ([catboost_acid_finalized](CBTrees/models/catboost_acid_finalized.pkl) and [catboost_base_finalized](CBTrees/models/catboost_base_finalized.pkl) ) trained on the whole dataset.

[CBTrees/calc_feat.py](CBTrees/KaML-CBtree.py) python script for feature calculation.

## KaML-GAT

[KaML-GAT/train_val_test](KaML-GAT/train_val_test) contains the datasets for training, validation and test for the 20 indenpendent splits. (exptAAfB_train/validation.csv, AA: AA_th split ID, B: B_th fold)

[KaML-GAT/model_inputs](KaML-GAT/model_inputs) contains the input files (constructured graph for each residue) for training, validation and test for the 20 indenpendent splits

[KaML-GAT/train_model.sh](KaML-GAT/train_model.sh) calls the train_model.py for training the model. Usage: bash train_model.sh

[KaML-GAT/models](KaML-GAT/models) trained_models.tar.gz contains all the trained models. model_training_results.tar.gz conatins the training recodings (*_traindtl.csv contains the training predictions, *_valdtl.csv contains the validation predictions, *_predictions contains the test predictions. *.training contaisn the loss for each epoch. ana_split0-19.ipynb is the analysis script (including convert the dpka preditions to the dpka before normalization, convert dpka to pka, ensemble creating etc).). 
