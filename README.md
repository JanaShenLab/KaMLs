# KaMLs for Protein Ionization State Predictions: Are Trees All You Need?

Machine Learning models for pKa predictions of amino acid side-chains.

## KaML-Trees

All training and test data splits as well as pretrained KaML-CBtree models.

[CBTrees/train_test_splits](CBTrees/train_test_split) contains the train/test sets for each of the 20 splits

[CBTrees/models](CBTrees/models) contains the CBTree models for each of the 20 test splits as well ass the finalized models  ([catboost_acid_finalized](CBTrees/models/catboost_acid_finalized.pkl) and [catboost_base_finalized](CBTrees/models/catboost_base_finalized.pkl) ) trained on the whole dataset.

[CBTrees/KaML-CBtree.py](CBTrees/KaML-CBtree.py) end-to-end prediction script

### Installation
### Usage

## KaML-GAT

[KaML-GAT/train_val_test](KaML-GAT/train_val_test) contains the datasets for training, validation and test for the 20 indenpendent splits. (exptAAfB_train/validation.csv, AA: AA_th split ID, B: B_th fold)

[KaML-GAT/model_inputs](KaML-GAT/model_inputs) contains the input files (constructured graph for each residue) for training, validation and test for the 20 indenpendent splits

[KaML-GAT/train_model.sh](KaML-GAT/train_model.sh) calls the train_model.py for training the model. Usage: bash train_model.sh

[KaML-GAT/models](KaML-GAT/models) trained_models.tar.gz contains all the trained models. model_training_results.tar.gz conatins the training recodings (*_traindtl.csv contains the training predictions, *_valdtl.csv contains the validation predictions, *_predictions contains the test predictions. *.training contaisn the loss for each epoch. ana_split0-19.ipynb is the analysis script (including convert the dpka preditions to the dpka before normalization, convert dpka to pka, ensemble creating etc).). 


## References

If you use KaML models in your research, please cite

### KaML

Shen M, ... 

### RIDA

Dayhoff GW II, Uversky VN. Rapid prediction and analysis of protein intrinsic disorder. Protein Science. 2022; 31(12):e4496. https://doi.org/10.1002/pro.4496


### DSSP
Kabsch W, Sander C (1983). "Dictionary of protein secondary structure: pattern recognition of hydrogen-bonded and geometrical features". Biopolymers. 22 (12): 2577–637. doi:10.1002/bip.360221211. PMID 6667333. S2CID 29185760

We included a precompiled binary (Linux) of DSSP which is provided under the Boost license:

Boost Software License - Version 1.0 - August 17th, 2003 Permission is hereby granted, free of charge, to any person or organization obtaining a copy of the software and accompanying documentation covered by this license (the "Software") to use, reproduce, display, distribute, execute, and transmit the Software, and to prepare derivative works of the Software, and to permit third-parties to whom the Software is furnished to do so, all subject to the following:

The copyright notices in the Software and this entire statement, including the above license grant, this restriction and the following disclaimer, must be included in all copies of the Software, in whole or in part, and all derivative works of the Software, unless such copies or derivative works are solely in the form of machine-executable object code generated by a source language processor.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. 



