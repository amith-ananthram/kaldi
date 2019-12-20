#!/bin/bash

# add compiled Kaldi executables to the path
. ./path.sh
. ./autoencoder_settings.sh

# hard fail on errors in the script
set -e

stage=placeholder
discriminator_model="seven_layers_with_sil_250"

. ./utils/parse_options.sh

# this should be set to our underlying feature dimensionality
# (ie # of MFCC + pitch features, so 33); this will be the 
# dimensionality of our encoded frames in latent space
NUM_FEAT_DIMENSIONS=33

# this should be the number of target emotions
# we're mapping to in MELD (for our updated output layer)
NUM_TARGET_DIMENSIONS=5

# set up expected input directory structure,
# copy reference model and source data for training sets
if [ $stage -eq 0 ]; then
	echo "Stage 0: start"
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
	cp models/${discriminator_model}.raw $MODEL_INPUT_DIR/$BASE_REFERENCE_MODEL

	# copy the MELD and IEMOCAP features, labels, predictions for the specified discriminator
	cp meld/outputs/data/all_meld/wav.scp $DATA_INPUT_DIR/meld_wav.scp
	cp meld/outputs/data/all_meld/utt2spk $DATA_INPUT_DIR/meld_utt2spk
	cp meld/predictions/${discriminator_model}_prediction.ark $DATA_INPUT_DIR/meld_predictions.ark

	cp iemocap/all_iemocap/wav.scp $DATA_INPUT_DIR/iemocap_wav.scp
	cp iemocap/all_iemocap/utt2spk $DATA_INPUT_DIR/iemocap_utt2spk
	cp iemocap/predictions/${discriminator_model}_prediction.ark $DATA_INPUT_DIR/iemocap_predictions.ark

	echo "Stage 0: end"
fi

# convert the reference model (the trained emotion detector) into an autoencoder 
# (we use the emotion detector as its final pinned layers to guide training)
if [ $stage -eq 1 ]; then
	echo "Stage 1: start"
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
		input dim=${input_dim} name=input
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

	# we'll use this to extract emotion-converted frames from the model
	echo "output-node name=output input=tdnn-1.affine" > $MODEL_OUTPUT_DIR/extract.config

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

	nnet3-info $MODEL_OUTPUT_DIR/$MODIFIED_REFERENCE_MODEL		
	echo "Stage 1: end"
fi

# select training examples from MELD and IEMOCAP based on
# how well the detector classifies them (choose only high accuracy examples)
if [ $stage -eq 2 ]; then
	generate_emotion_conversion_inputs.py $DATA_INPUT_DIR $DATA_OUTPUT_DIR
	utils/utt2spk_to_spk2utt.pl $DATA_OUTPUT_DIR/utt2spk > $DATA_OUTPUT_DIR/spk2utt
fi

# generate MFCC and pitch features for our training examples
if [ $stage -eq 3 ]; then
	echo "Stage 3: start"
	# Make MFCCs and compute the energy-based VAD for each dataset
	steps/make_mfcc_pitch.sh --write-utt2num-frames true --mfcc-config conf/mfcc.conf --pitch-config conf/pitch.conf --nj 40 --cmd "$train_cmd" \
		$DATA_OUTPUT_DIR ${BASE_DIR}/exp/make_mfcc ${BASE_DIR}/mfcc
	utils/fix_data_dir.sh $DATA_OUTPUT_DIR
  	echo "Stage 3: end"
fi
