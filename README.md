# KaMLs for Protein Ionization State Predictions: Are Trees All You Need?

Machine Learning models for pKa predictions of amino acid side-chains.

## KaML-Trees
### Data
All training and test data splits as well as pretrained KaML-CBtree models.

[KaML-CBTrees/train_test_splits](KaML-CBTrees/train_test_split) contains the train/test sets for each of the 20 splits

[KaML-CBTrees/models](KaML-CBTrees/models) contains the 20 models for evaluation as well as the finalized models  ([catboost_acid_finalized](CBTrees/models/catboost_acid_finalized.pkl) and [catboost_base_finalized](CBTrees/models/catboost_base_finalized.pkl) ) trained on the whole dataset.

### Making predictions
[KaML-CBTrees/KaML-CBtree.py](KaML-CBTrees/KaML-CBtree.py) end-to-end prediction script. This script takes a PDB file as input and will find all Asp, Glu, His, Cys, Lys, and Tyr residues, calculate tree model features for each residue, predict their pKa values using the finalized KaML-CBtree models, and save the results in a csv file. 

#### Installation

1. Clone repository.

2. Install requirements:

  * 3.0 < python < 3.12 (At time of writing this, pycaret does not work with python 3.12)

  * ``` pip install pycaret ```

  * ``` pip install Biopandas ```

  * ``` pip install catboost ```

#### Usage

```  python KaML-CBtree.py path/to/input.pdb ```

will generate a new file input.csv in the current working directory with residues in the first column and predicted pKa values in the second. 

#### Notes 
 * At the moment the code only works with single chain PDB files without missing atoms.
 * KaML-CBtree.py depends on features.py. Relative paths to rida and dssp are hard-coded in features.py. Relative paths to the model files are hard-coded in KaML-CBtree.py.

KaML-CBtrees uses the follwing software to calculate the features:
 * RIDA:
 Dayhoff GW II, Uversky VN. Rapid prediction and analysis of protein intrinsic disorder. Protein Science. 2022; 31(12):e4496. https://doi.org/10.1002/pro.4496

 * DSSP:
  Kabsch W, Sander C (1983). "Dictionary of protein secondary structure: pattern recognition of hydrogen-bonded and geometrical features". Biopolymers. 22 (12): 2577–637. https://doi:10.1002/bip.360221211. 

## KaML-GAT

[KaML-GAT/train_val_test](KaML-GAT/train_val_test) contains the datasets for training, validation and test for the 20 indenpendent splits. (exptAAfB_train/validation.csv, AA: AA_th split ID, B: B_th fold)

[KaML-GAT/model_inputs](KaML-GAT/model_inputs) contains the input files (constructured graph for each residue) for training, validation and test for the 20 indenpendent splits

[KaML-GAT/train_model.sh](KaML-GAT/train_model.sh) calls the train_model.py for training the model. Usage: bash train_model.sh

[KaML-GAT/models](KaML-GAT/models) trained_models.tar.gz contains all the trained models. model_training_results.tar.gz conatins the training recodings (*_traindtl.csv contains the training predictions, *_valdtl.csv contains the validation predictions, *_predictions contains the test predictions. *.training contaisn the loss for each epoch. ana_split0-19.ipynb is the analysis script (including convert the dpka preditions to the dpka before normalization, convert dpka to pka, ensemble creating etc).). 


## References

If you use KaML models in your research, please cite

### KaML

Shen M, ... 



