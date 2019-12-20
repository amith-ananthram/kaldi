#!/bin/bash

# add compiled Kaldi executables to the path
. ./path.sh
. ./autoencoder_settings.sh

# hard fail on errors in the script
set -e

stage=placeholder
discriminator_model="seven_layers_with_sil_250.raw"

. ./utils/parse_options.sh

# this should be set to our underlying feature dimensionality
# (ie # of MFCC + pitch features, so 33); this will be the 
# dimensionality of our encoded frames in latent space
NUM_FEAT_DIMENSIONS=33

# this should be the number of target emotions
# we're mapping to in MELD (for our updated output layer)
NUM_TARGET_DIMENSIONS=5

if [ $stage -eq 0 ]; then
	dirs=(
		$BASE_DIR
		$MODEL_INPUT_DIR 
		$MODEL_OUTPUT_DIR 
		$DATA_INPUT_DIR 
		$DATA_OUTPUT_DIR
	)
	for dir in "${dirs[@]}"
	do
		if [ ! -d $dir ]; then
			mkdir -p $dir
			sudo chmod -R 775 $dir
		fi
	done

	# copy the specified discriminator into our MODEL_INPUT_DIR
	cp models/$discriminator_model $MODEL_INPUT_DIR/$BASE_REFERENCE_MODEL
fi

if [ $stage -eq 1 ]; then
	# first, we delete everything that's already
	# in MODEL_OUTPUT_DIR (so we have a fresh start)
	rm -rf "$MODEL_OUTPUT_DIR/*"

	# then, we generate a new config for our modified
	# model which add a bunch of layers for the autoencoder
	
	input_dim=$(($NUM_FEAT_DIMENSIONS + 1))
	latent_dim=$NUM_FEAT_DIMENSIONS
	output_dim=$NUM_TARGET_DIMENSIONS

	max_chunk_size=10000
	min_chunk_size=25
	mkdir -p $MODEL_OUTPUT_DIR/configs
	cat <<-EOF > $MODEL_OUTPUT_DIR/configs/modified_nnet.xconfig
		# please note that it is important to have input layer with the name=input

		# autoencoder layers
		input dim=${feat_dim} name=input
		relu-batchnorm-layer name=tdnn-3 dim=1024 input=Append(-2,-1,0,1,2)
  		relu-batchnorm-layer name=tdnn-2 dim=512 input=Append(-1,2)
  		relu-batchnorm-layer name=tdnn-1 dim=${latent_dim} input=Append(-3,3)

		# below are the layers from our pre-trained emotion discriminator
		# (left unchanged so we reuse their weights -- learning rates are 0 below)
		relu-batchnorm-layer name=tdnn1 input=Append(-2,-1,0,1,2) dim=512
		relu-batchnorm-layer name=tdnn2 input=Append(-2,0,2) dim=512
		relu-batchnorm-layer name=tdnn3 input=Append(-3,0,3) dim=512
		relu-batchnorm-layer name=tdnn4 dim=512
		relu-batchnorm-layer name=tdnn5 dim=1500

		# The stats pooling layer. Layers after this are segment-level.
		# In the config below, the first and last argument (0, and ${max_chunk_size})
		# means that we pool over an input segment starting at frame 0
		# and ending at frame ${max_chunk_size} or earlier.  The other arguments (1:1)
		# mean that no subsampling is performed.
		stats-layer name=stats config=mean+stddev(0:1:1:${max_chunk_size})
		relu-batchnorm-layer name=tdnn6 dim=512 input=stats
		relu-batchnorm-layer name=tdnn7 dim=512

		output-layer name=output include-log-softmax=true dim=${output_dim}
	EOF

	steps/nnet3/xconfig_to_configs.py \
		--xconfig-file $MODEL_OUTPUT_DIR/configs/modified_nnet.xconfig \
		--config-dir $MODEL_OUTPUT_DIR/configs/
	cp $MODEL_OUTPUT_DIR/configs/final.config $MODEL_OUTPUT_DIR/modified_nnet.config

	# now, with our updated target config in hand, we copy our
	# reference model, modifying its final output layer and setting
	# the learning rates for the first 6 layers to 0
	nnet3-copy \
		--nnet-config="$MODEL_OUTPUT_DIR/modified_nnet.config" \
		--edits="
			set-learning-rate name=stats* learning-rate=0; \
			set-learning-rate name=tdnn1* learning-rate=0; \
			set-learning-rate name=tdnn2* learning-rate=0; \
			set-learning-rate name=tdnn3* learning-rate=0; \
			set-learning-rate name=tdnn4* learning-rate=0; \
			set-learning-rate name=tdnn5* learning-rate=0; \
			set-learning-rate name=tdnn6* learning-rate=0; \
			set-learning-rate name=tdnn7* learning-rate=0; \
			set-learning-rate name=output* learning-rate=0;" \
		"$MODEL_INPUT_DIR/$BASE_REFERENCE_MODEL" \
		"$MODEL_OUTPUT_DIR/$MODIFIED_REFERENCE_MODEL" || exit 1;
fi