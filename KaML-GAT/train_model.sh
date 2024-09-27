#! /bin/bash
#usage bash train_model.sh model_name n_expt

model="$1"
for n_expt in {10..19};
do
for fold in {0..9};
do
sed "s/%n_expt%/$n_expt/g" $model > tmp.py
sed "s/%n_fold%/$fold/g" tmp.py > tmp2.py
python tmp2.py
done
done
