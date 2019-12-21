#!/bin/bash

# author: aa4461, adapted from v2/run.sh

. ./cmd.sh
. ./path.sh
. nnet-emotion/detector/training/meld_settings.sh

set -e

stage=0
train_stage=-1
remove_sil=true
min_num_frames=250

. ./utils/parse_options.sh

root="${BASE_DIR}"
mfccdir="$root/mfcc"
vaddir="$root/mfcc"
nnet_dir="$root/exp/xvector_nnet_1a"
data_dir="${DATA_OUTPUT_DIR}"

musan_root=corpora/musan

# make expected directory structure (if it doesn't already exist)
if [ $stage -le 0 ]; then
  echo "stage 0: start"

  nnet-emotion/detector/training/prepare_meld.sh --stage 0

  echo "stage 0: end"
fi

# prepare reference model (safe to rerun; it will
# just over-write any modified reference model)
if [ $stage -le 1 ]; then
  echo "stage 1: start"

  nnet-emotion/detector/training/prepare_meld.sh --stage 1

  echo "stage 1: end"
fi

# prepare input data: utt2spk, wav.scp (safe to rerun; it will
# just over-write any existing generated input files)
if [ $stage -le 2 ]; then
  echo "stage 2: start"
  nnet-emotion/detector/training/prepare_meld.sh --stage 2
  echo "stage 2: end"
fi

# if we end up training off the combination of MELD + EmoVoxCeleb,
# we need to combine the corpora here using utils/combine_data.sh

if [ $stage -le 3 ]; then
  echo "stage 3: start"
  # Make MFCCs and compute the energy-based VAD for each dataset
  steps/make_mfcc_pitch.sh --write-utt2num-frames true --mfcc-config conf/mfcc.conf --pitch-config conf/pitch.conf --nj 40 --cmd "$train_cmd" \
      ${DATA_OUTPUT_COMBINED_DIR} ${root}/exp/make_mfcc $mfccdir
  utils/fix_data_dir.sh ${DATA_OUTPUT_COMBINED_DIR}
  sid/compute_vad_decision.sh --nj 40 --cmd "$train_cmd" \
    ${DATA_OUTPUT_COMBINED_DIR} ${root}/exp/make_vad $vaddir
  utils/fix_data_dir.sh ${DATA_OUTPUT_COMBINED_DIR}
  echo "stage 3: end"
fi

# In this section, we augment the MELD data with reverberation,
# noise, music, and babble, and combine it with the clean data.
if [ $stage -le 4 ]; then
  echo "stage 4: start"
  frame_shift=0.01
  awk -v frame_shift=$frame_shift '{print $1, $2*frame_shift;}' ${DATA_OUTPUT_COMBINED_DIR}/utt2num_frames > ${DATA_OUTPUT_COMBINED_DIR}/reco2dur

  if [ ! -d "RIRS_NOISES" ]; then
    # Download the package that includes the real RIRs, simulated RIRs, isotropic noises and point-source noises
    wget --no-check-certificate http://www.openslr.org/resources/28/rirs_noises.zip
    unzip rirs_noises.zip
  fi

  # Make a version with reverberated speech
  rvb_opts=()
  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/smallroom/rir_list")
  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/mediumroom/rir_list")

  # Make a reverberated version of the MELD list.  Note that we don't add any
  # additive noise here.
  steps/data/reverberate_data_dir.py \
    "${rvb_opts[@]}" \
    --speech-rvb-probability 1 \
    --pointsource-noise-addition-probability 0 \
    --isotropic-noise-addition-probability 0 \
    --num-replications 1 \
    --source-sampling-rate 16000 \
    ${DATA_OUTPUT_COMBINED_DIR} ${data_dir}/train_reverb
  cp ${DATA_OUTPUT_COMBINED_DIR}/vad.scp ${data_dir}/train_reverb/
  utils/copy_data_dir.sh --utt-suffix "-reverb" ${data_dir}/train_reverb ${data_dir}/train_reverb.new
  rm -rf ${data_dir}/train_reverb
  mv ${data_dir}/train_reverb.new ${data_dir}/train_reverb

  # Prepare the MUSAN corpus, which consists of music, speech, and noise
  # suitable for augmentation.
  steps/data/make_musan.sh --sampling-rate 16000 $musan_root ${data_dir}

  # Get the duration of the MUSAN recordings.  This will be used by the
  # script augment_data_dir.py.
  for name in speech noise music; do
    utils/data/get_utt2dur.sh ${data_dir}/musan_${name}
    mv ${data_dir}/musan_${name}/utt2dur ${data_dir}/musan_${name}/reco2dur
  done

  # Augment with musan_noise
  steps/data/augment_data_dir.py --utt-suffix "noise" --fg-interval 1 --fg-snrs "15:10:5:0" --fg-noise-dir ${data_dir}/musan_noise ${DATA_OUTPUT_COMBINED_DIR} ${data_dir}/train_noise
  # Augment with musan_music
  steps/data/augment_data_dir.py --utt-suffix "music" --bg-snrs "15:10:8:5" --num-bg-noises "1" --bg-noise-dir ${data_dir}/musan_music ${DATA_OUTPUT_COMBINED_DIR} ${data_dir}/train_music
  # Augment with musan_speech
  steps/data/augment_data_dir.py --utt-suffix "babble" --bg-snrs "20:17:15:13" --num-bg-noises "3:4:5:6:7" --bg-noise-dir ${data_dir}/musan_speech ${DATA_OUTPUT_COMBINED_DIR} ${data_dir}/train_babble

  # Combine reverb, noise, music, and babble into one directory.
  utils/combine_data.sh ${data_dir}/train_aug ${data_dir}/train_reverb ${data_dir}/train_noise ${data_dir}/train_music ${data_dir}/train_babble
  echo "stage 4: end"
fi

if [ $stage -le 5 ]; then
  echo "stage 5: start"
  # Take a random subset of the augmentations
  utils/subset_data_dir.sh ${data_dir}/train_aug 10000 ${data_dir}/train_aug_1m
  utils/fix_data_dir.sh ${data_dir}/train_aug_1m

  # Make MFCCs for the augmented data.  Note that we do not compute a new
  # vad.scp file here.  Instead, we use the vad.scp from the clean version of
  # the list.
  steps/make_mfcc_pitch.sh --mfcc-config conf/mfcc.conf --pitch-config conf/pitch.conf --nj 40 --cmd "$train_cmd" \
    ${data_dir}/train_aug_1m ${root}/exp/make_mfcc $mfccdir

  # Combine the clean and augmented MELD list.  This is now roughly
  # double the size of the original clean list.
  utils/combine_data.sh ${data_dir}/train_combined ${data_dir}/train_aug_1m ${DATA_OUTPUT_COMBINED_DIR}
  echo "stage 5: end"
fi

# Now we prepare the features to generate examples for xvector training.
if [ $stage -le 6 ]; then
  echo "stage 6: start"
  # This script applies CMVN and removes nonspeech frames.  Note that this is somewhat
  # wasteful, as it roughly doubles the amount of training data on disk.  After
  # creating training examples, this can be removed.
  rm -rf ${data_dir}/train_combined_no_sil
  if [ $remove_sil ]; then
	local/nnet3/xvector/prepare_feats_for_egs.sh --nj 40 --cmd "$train_cmd" \
		${data_dir}/train_combined ${data_dir}/train_combined_no_sil ${root}/exp/train_combined_no_sil
	utils/fix_data_dir.sh ${data_dir}/train_combined_no_sil
  else 
  	cp -r ${data_dir}/train_combined ${data_dir}/train_combined_no_sil
  fi
  echo "stage 6: end"
fi

# ./run.sh does a bunch of filtering of utterances by speakers
# that are too infrequent -- we skip that here speaker=emotion label 
if [ $stage -le 7 ]; then
  echo "stage 7: start"
  # Now, we need to remove features that are too short after removing silence
  # frames.  We want atleast ~1s (100 frames) per utterance. (note this is smaller than in v2/run.sh)
  min_len=$min_num_frames
  mv ${data_dir}/train_combined_no_sil/utt2num_frames ${data_dir}/train_combined_no_sil/utt2num_frames.bak
  awk -v min_len=${min_len} '$2 > min_len {print $1, $2}' ${data_dir}/train_combined_no_sil/utt2num_frames.bak > ${data_dir}/train_combined_no_sil/utt2num_frames
  utils/filter_scp.pl ${data_dir}/train_combined_no_sil/utt2num_frames ${data_dir}/train_combined_no_sil/utt2spk > ${data_dir}/train_combined_no_sil/utt2spk.new
  mv ${data_dir}/train_combined_no_sil/utt2spk.new ${data_dir}/train_combined_no_sil/utt2spk
  utils/fix_data_dir.sh ${data_dir}/train_combined_no_sil

  echo "stage 7: end"
fi

if [ $stage -le 8 ]; then
  echo "stage 8: start"

  sid/nnet3/xvector/get_egs.sh --cmd "$train_cmd" \
    --nj 8 \
    --stage 0 \
    --frames-per-iter 25000000 \
    --frames-per-iter-diagnostic 100000 \
    --min-frames-per-chunk 50 \
    --max-frames-per-chunk $min_num_frames \
    --num-diagnostic-archives 3 \
    --num-repeats 500 \
    ${data_dir}/train_combined_no_sil $nnet_dir/egs

  echo "stage 8: end"
fi

dropout_schedule='0,0@0.20,0.1@0.50,0'
srand=123
if [ $stage -le 9 ]; then
  steps/nnet3/train_raw_dnn.py --stage=$train_stage \
    --cmd="$train_cmd" \
    --trainer.input-model "${MODEL_OUTPUT_DIR}/${MODIFIED_REFERENCE_MODEL}" \
    --trainer.optimization.proportional-shrink 10 \
    --trainer.optimization.momentum=0.5 \
    --trainer.optimization.num-jobs-initial=1 \
    --trainer.optimization.num-jobs-final=3 \
    --trainer.optimization.initial-effective-lrate=0.001 \
    --trainer.optimization.final-effective-lrate=0.0001 \
    --trainer.optimization.minibatch-size=64 \
    --trainer.srand=$srand \
    --trainer.max-param-change=2 \
    --trainer.num-epochs=6 \
    --trainer.dropout-schedule="$dropout_schedule" \
    --trainer.shuffle-buffer-size=1000 \
    --egs.frames-per-eg=1 \
    --egs.dir=$nnet_dir/egs \
    --cleanup.remove-egs false \
    --cleanup.preserve-model-interval=10 \
    --use-gpu=wait \
    --dir=$nnet_dir  || exit 1;
fi
