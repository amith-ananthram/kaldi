#!/bin/bash

# author: aa4461, adapted from v2/run.sh

. ./cmd.sh
. ./path.sh
. nnet-emotion/detector/training/finetune_settings.sh

set -e

stage=0

# eg cremad,iemocap(1|2|3|4|5)
train_corpora=placeholder
# eg, voice|all,
# for cremad: "voice/multi|any/all"
# for iemocap, "improv/script/all"
# should be same same length as train_corpora
train_corpora_config=placeholder

num_target_dimensions=placeholder
# "select" or "collapse"
target_emotions_mode=placeholder
# if select, comma separated list of emotions to use, eg: ang,hap,sad,neu (drops others)
# if collapse, parenthesis wrapped comma separated list of groupings, eg: ang/sad,dis/fear
target_emotions_config=placeholder

include_noise=true
# -1 will randomly sample the same number of noisy
# samples as exist across the corpora already
num_noisy_samples=-1
remove_sil=false

# controls the size of example generation used
# for nnet training (filters out utterances shorter
# than min_num_frames FYI!)
min_num_frames=150
max_num_frames=150

# these command line options allow specifying more 
# / fewer layers in the fine-tuned model and different 
# learning rates for the original layers
num_layers=7
first_six_lr=0
dropout=placeholder
epochs=6

# if training fails, can restart with this parameter
train_stage=-1

base=placeholder

# used to name experiment output, defaults to runtime
variant=$(date '+%Y%m%d%H%M%S')

. ./utils/parse_options.sh

root="${BASE_DIR}"
mfccdir="$root/mfcc"
vaddir="$root/mfcc"
nnet_dir="$root/exp/xvector_nnet_1a"

function log {
	echo "$(date) stage=$stage $1"
}

function log_stage_start {
	log "START STAGE $stage: ($stage_details)"
}

function log_stage_end {
	log "END STAGE $stage: ($stage_details)"
}

function error {
	echo "$(date) stage=$stage $1" 1>&2
	exit 1
}

log "Finetuning base VoxCeleb model with:"
log "Data config:"
log "train_corpora=$train_corpora"
log "train_corpora_config=$train_corpora_config"
log "num_target_dimensions=$num_target_dimensions"
log "target_emotions_mode=$target_emotions_mode"
log "target_emotions_config=$target_emotions_config"
log "include_noise=$include_noise"
log "num_noisy_samples=$num_noisy_samples"
log "remove_sil=$remove_sil"
log "min_num_frames=$min_num_frames"
log "Optimization config:"
log "train_stage=$train_stage"
log "num_layers=$num_layers"
log "first_six_lr=$first_six_lr"
log "dropout=$dropout"
log "epochs=$epochs"

if [ $base -eq 1 ]; then 
	base_model=vox2_base1.raw
	mfcc_conf=conf/mfcc.conf
	num_input_dimensions=33
	log "using Vox2 model 1..."
elif [ $base -eq 2 ]; then
	base_model=vo2_base2.raw
	mfcc_conf=conf/tedlium_mfcc.conf
	num_input_dimension=43
	log "using Vox2 model 2..."
else
	error "Unsupported base=$base"
fi

if [ $stage -le 0 ]; then
	stage_details="making directory structure"
	log_stage_start

	# a clean start
	rm -rf $BASE_DIR

	dirs=(
		$BASE_DIR
		$INPUT_DIR
		$DATA_OUTPUT_DIR
		$MODEL_INPUT_DIR
		$MODEL_OUTPUT_DIR
		$DATA_OUTPUT_DIR
		$DATA_OUTPUT_COMBINED_DIR
	)
	for dir in "${dirs[@]}"
	do
		if [ ! -d $dir ]; then
			mkdir -p $dir
			chmod -R 775 $dir
		fi
	done
	cp $base_model $MODEL_INPUT_DIR
	log_stage_end
fi

if [ $stage -le 1 ]; then
	stage_details="preparing model"
	log_stage_start

	# first, we delete everything that's already
	# in MODEL_OUTPUT_DIR (so we have a fresh start)
	rm -rf "$MODEL_OUTPUT_DIR/*"

	if [ $num_layers -eq 7 ]; then 
		additional_layers=''
	elif [ $num_layers -eq 8 ]; then
		additional_layers='relu-batchnorm-layer name=tdnn8 dim=512'
	elif [ $num_layers -eq 9 ]; then 
		additional_layers=<<-ADDITIONAL_LAYERS
			relu-batchnorm-layer name=tdnn8 dim=512
			relu-batchnorm-layer name=tdnn9 dim=256
		ADDITIONAL_LAYERS
	else 
		error "Unsupported num_layers=$num_layers"
	fi

	if [ $first_six_lr -lt 0 ]; then 
		error "Unsupported first_six_lr=$first_six_lr"
	fi

	# then, we generate a new config for our modified
	# model which modifies the output dimensionality of
	# the final layer (from # of speakers in vox2 to # of emotions)

	# 
	# START: copied with minimal modification from run_xvector.sh
	#
	min_chunk_size=25
	max_chunk_size=10000

	mkdir -p $MODEL_OUTPUT_DIR/configs
	cat <<-EOF > $MODEL_OUTPUT_DIR/configs/modified_nnet.xconfig
		# please note that it is important to have input layer with the name=input

		# The frame-level layers
		input dim=${num_input_dimensions} name=input
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
		${additional_layers}
		output-layer name=output include-log-softmax=true dim=${num_target_dimensions}
	EOF

	steps/nnet3/xconfig_to_configs.py \
		--xconfig-file $MODEL_OUTPUT_DIR/configs/modified_nnet.xconfig \
		--config-dir $MODEL_OUTPUT_DIR/configs/
	cp $MODEL_OUTPUT_DIR/configs/final.config $MODEL_OUTPUT_DIR/modified_nnet.config

	echo "$min_chunk_size" > $MODEL_OUTPUT_DIR/min_chunk_size
	echo "$max_chunk_size" > $MODEL_OUTPUT_DIR/max_chunk_size
	
	#
	# END: copied with minimal modification from run_xvector.sh
	#

	# now, with our updated target config in hand, we copy our
	# reference model, modifying its final output layer and setting
	# the learning rates for the first 6 layers to $first_six_lr
	nnet3-copy \
		--nnet-config="$MODEL_OUTPUT_DIR/modified_nnet.config" \
		--edits="set-learning-rate name=input* learning-rate=$first_six_lr; \
			set-learning-rate name=stats* learning-rate=$first_six_lr; \
			set-learning-rate name=tdnn1* learning-rate=$first_six_lr; \
			set-learning-rate name=tdnn2* learning-rate=$first_six_lr; \
			set-learning-rate name=tdnn3* learning-rate=$first_six_lr; \
			set-learning-rate name=tdnn4* learning-rate=$first_six_lr; \
			set-learning-rate name=tdnn5* learning-rate=$first_six_lr; \
			set-learning-rate name=tdnn6* learning-rate=$first_six_lr;" \
		"$MODEL_INPUT_DIR/$BASE_REFERENCE_MODEL" \
		"$MODEL_OUTPUT_DIR/$MODIFIED_REFERENCE_MODEL" || exit 1;
	
	log_stage_end
fi

if [ $stage -le 2 ]; then
	stage_details="generating various input files (utt2spk, spk2utt, etc)"
	log_stage_start

	nnet-emotion/detector/training/generate_corpora_inputs.py \
		--train_corpora $train_corpora \
		--train_corpora_config $train_corpora_config \
		--target_emotions_mode $target_emotions_mode \
		--target_emotions_config $target_emotions_config \
		--output_dir $DATA_OUTPUT_COMBINED_DIR

	# call to a single python directory which generates the various files for train_corpora
	utils/utt2spk_to_spk2utt.pl "$DATA_OUTPUT_COMBINED_DIR/utt2spk" > "$DATA_OUTPUT_COMBINED_DIR/spk2utt"

	log_stage_end
fi

if [ $stage -le 3 ]; then 
	stage_details="extracting MFCCs and VAD"
	log_stage_start

	steps/make_mfcc_pitch.sh --write-utt2num-frames true --mfcc-config $mfcc_conf --pitch-config conf/pitch.conf --nj 40 --cmd "$train_cmd" \
		${DATA_OUTPUT_COMBINED_DIR} ${root}/exp/make_mfcc $mfccdir
	utils/fix_data_dir.sh ${DATA_OUTPUT_COMBINED_DIR}
	sid/compute_vad_decision.sh --nj 40 --cmd "$train_cmd" \
	  ${DATA_OUTPUT_COMBINED_DIR} ${root}/exp/make_vad $vaddir
	utils/fix_data_dir.sh ${DATA_OUTPUT_COMBINED_DIR}

	log_stage_end 
fi 

if [ $stage -le 4 ]; then 
	stage_details="adding noise"
	log_stage_start

	if $include_noise; then 
		frame_shift=0.01
		awk -v frame_shift=$frame_shift '{print $1, $2*frame_shift;}' ${DATA_OUTPUT_COMBINED_DIR}/utt2num_frames > ${DATA_OUTPUT_COMBINED_DIR}/reco2dur

		# Make a version with reverberated speech
		rvb_opts=()
		rvb_opts+=(--rir-set-parameters "0.5, $NOISE_DIR/simulated_rirs/smallroom/rir_list")
		rvb_opts+=(--rir-set-parameters "0.5, $NOISE_DIR/simulated_rirs/mediumroom/rir_list")

		# Make a reverberated version of the CremaD list.  Note that we don't add any
		# additive noise here.
		rm -rf ${DATA_OUTPUT_DIR}/train_reverb
		steps/data/reverberate_data_dir.py \
		  "${rvb_opts[@]}" \
		  --speech-rvb-probability 1 \
		  --pointsource-noise-addition-probability 0 \
		  --isotropic-noise-addition-probability 0 \
		  --num-replications 1 \
		  --source-sampling-rate 16000 \
		  ${DATA_OUTPUT_COMBINED_DIR} ${DATA_OUTPUT_DIR}/train_reverb
		cp ${DATA_OUTPUT_COMBINED_DIR}/vad.scp ${DATA_OUTPUT_DIR}/train_reverb/
		utils/copy_data_dir.sh --utt-suffix "-reverb" ${DATA_OUTPUT_DIR}/train_reverb ${DATA_OUTPUT_DIR}/train_reverb.new
		rm -rf ${DATA_OUTPUT_DIR}/train_reverb
		mv ${DATA_OUTPUT_DIR}/train_reverb.new ${DATA_OUTPUT_DIR}/train_reverb

		# Prepare the MUSAN corpus, which consists of music, speech, and noise
		# suitable for augmentation.
		steps/data/make_musan.sh --sampling-rate 16000 $MUSAN_DIR $DATA_OUTPUT_DIR

		# Get the duration of the MUSAN recordings.  This will be used by the
		# script augment_data_dir.py.
		for name in speech noise music; do
		  utils/data/get_utt2dur.sh ${DATA_OUTPUT_DIR}/musan_${name}
		  mv ${DATA_OUTPUT_DIR}/musan_${name}/utt2dur ${DATA_OUTPUT_DIR}/musan_${name}/reco2dur
		done

		# Augment with musan_noise
		steps/data/augment_data_dir.py --utt-suffix "noise" --fg-interval 1 --fg-snrs "15:10:5:0" --fg-noise-dir ${DATA_OUTPUT_DIR}/musan_noise ${DATA_OUTPUT_COMBINED_DIR} ${DATA_OUTPUT_DIR}/train_noise
		# Augment with musan_music
		steps/data/augment_data_dir.py --utt-suffix "music" --bg-snrs "15:10:8:5" --num-bg-noises "1" --bg-noise-dir ${DATA_OUTPUT_DIR}/musan_music ${DATA_OUTPUT_COMBINED_DIR} ${DATA_OUTPUT_DIR}/train_music
		# Augment with musan_speech
		steps/data/augment_data_dir.py --utt-suffix "babble" --bg-snrs "20:17:15:13" --num-bg-noises "3:4:5:6:7" --bg-noise-dir ${DATA_OUTPUT_DIR}/musan_speech ${DATA_OUTPUT_COMBINED_DIR} ${DATA_OUTPUT_DIR}/train_babble

		# Combine reverb, noise, music, and babble into one directory.
		utils/combine_data.sh ${DATA_OUTPUT_DIR}/train_aug ${DATA_OUTPUT_DIR}/train_reverb ${DATA_OUTPUT_DIR}/train_noise ${DATA_OUTPUT_DIR}/train_music ${DATA_OUTPUT_DIR}/train_babble	

		if [ $num_noisy_samples -eq -1 ]; then
			num_noisy_samples=$(wc -l $DATA_OUTPUT_COMBINED_DIR/utt2spk | awk '{ print $1 }')
		fi
		log "Randomly sampling $num_noisy_samples noise examples and extracting their MFCCs...."
		utils/subset_data_dir.sh ${DATA_OUTPUT_DIR}/train_aug $num_noisy_samples ${DATA_OUTPUT_DIR}/train_aug_sub
	    utils/fix_data_dir.sh ${DATA_OUTPUT_DIR}/train_aug_sub
		
		# Make MFCCs for the augmented data.  Note that we do not compute a new
    	# vad.scp file here.  Instead, we use the vad.scp from the clean version of
    	# the list.
    	steps/make_mfcc_pitch.sh --mfcc-config $mfcc_conf --pitch-config conf/pitch.conf --nj 40 --cmd "$train_cmd" \
      		${DATA_OUTPUT_DIR}/train_aug_sub ${root}/exp/make_mfcc $mfccdir
      	utils/combine_data.sh ${DATA_OUTPUT_DIR}/train_combined_temp ${DATA_OUTPUT_DIR}/train_aug_sub ${DATA_OUTPUT_COMBINED_DIR}
    	rm -rf ${DATA_OUTPUT_COMBINED_DIR}
    	mv ${DATA_OUTPUT_DIR}/train_combined_temp ${DATA_OUTPUT_COMBINED_DIR}
    	rm -rf ${DATA_OUTPUT_DIR}/train_combined_temp
	else
		log "include_noise=$include_noise, doing nothing."
	fi

	log_stage_end
fi

if [ $stage -le 5 ]; then 
	stage_details="removing silence"
	log_stage_start

	rm -rf ${DATA_OUTPUT_DIR}/train_combined_no_sil
	local/nnet3/xvector/prepare_feats_for_egs.sh --nj 40 --remove_sil $remove_sil --cmd "$train_cmd" \
  			${DATA_OUTPUT_COMBINED_DIR} ${DATA_OUTPUT_DIR}/train_combined_no_sil ${root}/exp/train_combined_no_sil
  		utils/fix_data_dir.sh ${DATA_OUTPUT_DIR}/train_combined_no_sil

	log_stage_end
fi

if [ $stage -le 6 ]; then 
	stage_details="filtering utterances < $max_num_frames"
	log_stage_start

	# Now, we need to remove features that are too short.
	min_len=$max_num_frames
	mv ${DATA_OUTPUT_DIR}/train_combined_no_sil/utt2num_frames ${DATA_OUTPUT_DIR}/train_combined_no_sil/utt2num_frames.bak
	awk -v min_len=${min_len} '$2 > min_len {print $1, $2}' ${DATA_OUTPUT_DIR}/train_combined_no_sil/utt2num_frames.bak > ${DATA_OUTPUT_DIR}/train_combined_no_sil/utt2num_frames
	utils/filter_scp.pl ${DATA_OUTPUT_DIR}/train_combined_no_sil/utt2num_frames ${DATA_OUTPUT_DIR}/train_combined_no_sil/utt2spk > ${DATA_OUTPUT_DIR}/train_combined_no_sil/utt2spk.new
	mv ${DATA_OUTPUT_DIR}/train_combined_no_sil/utt2spk.new ${DATA_OUTPUT_DIR}/train_combined_no_sil/utt2spk
	utils/fix_data_dir.sh ${DATA_OUTPUT_DIR}/train_combined_no_sil

	log_stage_end
fi

if [ $stage -le 7 ]; then
	stage_details="generating training examples"
	log_stage_start

	rm -rf $nnet_dir/egs
	sid/nnet3/xvector/get_egs.sh --cmd "$train_cmd" \
	--nj 8 \
	--stage 0 \
	--frames-per-iter 30000000 \
	--frames-per-iter-diagnostic 100000 \
	--min-frames-per-chunk $min_num_frames \
	--max-frames-per-chunk $max_num_frames \
	--num-diagnostic-archives 3 \
	--num-repeats 500 \
	${DATA_OUTPUT_DIR}/train_combined_no_sil $nnet_dir/egs

	log_stage_end
fi

srand=123
if [ $stage -le 8 ]; then
	stage_details="training"
	log_stage_start

	steps/nnet3/train_raw_dnn.py --stage=$train_stage \
		--cmd="$train_cmd" \
		--trainer.input-model "${MODEL_OUTPUT_DIR}/${MODIFIED_REFERENCE_MODEL}" \
		--trainer.optimization.proportional-shrink 10 \
		--trainer.optimization.momentum=0.5 \
		--trainer.optimization.num-jobs-initial=1 \
		--trainer.optimization.num-jobs-final=1 \
		--trainer.optimization.initial-effective-lrate=0.001 \
		--trainer.optimization.final-effective-lrate=0.0001 \
		--trainer.optimization.minibatch-size=64 \
		--trainer.srand=$srand \
		--trainer.max-param-change=2 \
		--trainer.num-epochs=$epochs \
		--trainer.dropout-schedule="$dropout" \
		--trainer.shuffle-buffer-size=1000 \
		--egs.frames-per-eg=1 \
		--egs.dir=$nnet_dir/egs \
		--cleanup.remove-egs false \
		--cleanup.preserve-model-interval=10 \
		--use-gpu=wait \
		--dir=$nnet_dir  || exit 1;

	variant_dir=models/$variant
	mkdir $variant_dir
	mv train_log.txt $variant_dir
	mv $BASE_DIR/exp/xvector_nnet_1a/final.raw $variant_dir
	mv $BASE_DIR/exp/xvector_nnet_1a/accuracy.output.report $variant_dir
	
	log_stage_end
fi
