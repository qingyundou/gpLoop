#!/bin/bash

# module load miniconda/3
# conda init bash
source ~/.bashrc

conda activate gpLoop
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/qd212/.conda/envs/gpLoop/lib/

export CC=mpicc

lang=EN

work_dir=/home/qd212/models/gpLoop
cd $work_dir

direction=${1:-g2p}
mode=${2:-test}
exp_name=${3:-default}
echo $direction $mode

config=$work_dir/run/configs/${exp_name}.ini

echo config file from $config

if [ $direction = g2p ]; then
	if [ $mode = train ]; then
		LANGUAGE=$lang python train_g2p.py --config $config
	fi
	if [ $mode = test ]; then
		LANGUAGE=$lang python test_g2p.py --config $config --word PYTHON
	fi
fi

if [ $direction = p2g ]; then
	if [ $mode = train ]; then
		LANGUAGE=$lang python train_p2g.py --config $config
	fi
	if [ $mode = test ]; then
		LANGUAGE=$lang python test_p2g.py --config $config --word P.IH.TH.AH.N
	fi
fi

if [ $direction = accent ]; then
	LANGUAGE=$lang python add_accent.py --config $config
fi

# LANGUAGE=$lang python train_g2p.py

# LANGUAGE=$lang python train_p2g.py


# LANGUAGE=$lang python test_p2g.py --word P.IH.TH.AH.N