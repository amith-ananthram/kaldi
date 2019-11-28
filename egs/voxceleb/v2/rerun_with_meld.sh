#!/bin/bash
# Adapted from v2/run.sh (copyright information for whcih is below)
# Copyright   2017   Johns Hopkins University (Author: Daniel Garcia-Romero)
#             2017   Johns Hopkins University (Author: Daniel Povey)
#        2017-2018   David Snyder
#             2018   Ewald Enzinger
# Apache 2.0.
#
# See ../MELD_README.txt for more info on data required.

. ./cmd.sh
. ./path.sh
. ./meld_settings.sh

set -e

stage=0

. ./utils/parse_options.sh

root="${BASE_DIR}"
mfccdir="$root/mfcc"
vaddir="$root/mfcc"
nnet_dir="$root/exp/xvector_nnet_1a"
data_dir="${DATA_OUTPUT_DIR}"

musan_root=corpora/JHU/musan

# make expected directory structure (if it doesn't already exist)
if [ $stage -eq 0 ]; then
  prepare_meld.sh --stage 0
fi

# prepare reference model (safe to rerun; it will
# just over-write any modified reference model)
if [ $stage -eq 1 ]; then
  prepare_meld.sh --stage 1
fi

# prepare input data: utt2spk, wav.scp (safe to rerun; it will
# just over-write any existing generated input files)
if [ $stage -eq 2 ]; then
  prepare_meld.sh --stage 2
fi

# if we end up training off the combination of MELD + EmoVoxCeleb,
# we need to combine the corpora here using utils/combine_data.sh

if [ $stage -eq 3 ]; then
  # Make MFCCs and compute the energy-based VAD for each dataset
  for name in train test; do
    steps/make_mfcc.sh --write-utt2num-frames true --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" \
      ${data_dir}/${name} ${root}/exp/make_mfcc $mfccdir
    utils/fix_data_dir.sh ${data_dir}/${name}
    sid/compute_vad_decision.sh --nj 40 --cmd "$train_cmd" \
      ${data_dir}/${name} ${root}/exp/make_vad $vaddir
    utils/fix_data_dir.sh ${data_dir}/${name}
  done
fi

# In this section, we augment the MELD data with reverberation,
# noise, music, and babble, and combine it with the clean data.
if [ $stage -eq 4 ]; then
  frame_shift=0.01
  awk -v frame_shift=$frame_shift '{print $1, $2*frame_shift;}' ${data_dir}/train/utt2num_frames > ${data_dir}/train/reco2dur

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
    ${data_dir}/train ${data_dir}/train_reverb
  cp ${data_dir}/train/vad.scp ${data_dir}/train_reverb/
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
  steps/data/augment_data_dir.py --utt-suffix "noise" --fg-interval 1 --fg-snrs "15:10:5:0" --fg-noise-dir "data/musan_noise" ${data_dir}/train ${data_dir}/train_noise
  # Augment with musan_music
  steps/data/augment_data_dir.py --utt-suffix "music" --bg-snrs "15:10:8:5" --num-bg-noises "1" --bg-noise-dir "data/musan_music" ${data_dir}/train ${data_dir}/train_music
  # Augment with musan_speech
  steps/data/augment_data_dir.py --utt-suffix "babble" --bg-snrs "20:17:15:13" --num-bg-noises "3:4:5:6:7" --bg-noise-dir "data/musan_speech" ${data_dir}/train ${data_dir}/train_babble

  # Combine reverb, noise, music, and babble into one directory.
  utils/combine_data.sh ${data_dir}/train_aug ${data_dir}/train_reverb ${data_dir}/train_noise ${data_dir}/train_music ${data_dir}/train_babble
fi

if [ $stage -eq 5 ]; then
  # Take a random subset of the augmentations
  utils/subset_data_dir.sh ${data_dir}/train_aug 10000 ${data_dir}/train_aug_1m
  utils/fix_data_dir.sh ${data_dir}/train_aug_1m

  # Make MFCCs for the augmented data.  Note that we do not compute a new
  # vad.scp file here.  Instead, we use the vad.scp from the clean version of
  # the list.
  steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" \
    ${data_dir}/train_aug_1m ${root}/exp/make_mfcc $mfccdir

  # Combine the clean and augmented MELD list.  This is now roughly
  # double the size of the original clean list.
  utils/combine_data.sh ${data_dir}/train_combined ${data_dir}/train_aug_1m ${data_dir}/train
fi

# Now we prepare the features to generate examples for xvector training.
if [ $stage -eq 6 ]; then
  # This script applies CMVN and removes nonspeech frames.  Note that this is somewhat
  # wasteful, as it roughly doubles the amount of training data on disk.  After
  # creating training examples, this can be removed.
  local/nnet3/xvector/prepare_feats_for_egs.sh --nj 2 --cmd "$train_cmd" \
    ${data_dir}/train_combined ${data_dir}/train_combined_no_sil ${root}/exp/train_combined_no_sil
  utils/fix_data_dir.sh ${data_dir}/train_combined_no_sil
fi

# ./run.sh does a bunch of filtering of utterances by speakers
# that are too infrequent -- we skip that here speaker=emotion label 
if [ $stage -eq 7 ]; then
  # Now, we need to remove features that are too short after removing silence
  # frames.  We want atleast ~1s (100 frames) per utterance. (note this is smaller than in v2/run.sh)
  min_len=100
  mv ${data_dir}/train_combined_no_sil/utt2num_frames ${data_dir}/train_combined_no_sil/utt2num_frames.bak
  awk -v min_len=${min_len} '$2 > min_len {print $1, $2}' ${data_dir}/train_combined_no_sil/utt2num_frames.bak > ${data_dir}/train_combined_no_sil/utt2num_frames
  utils/filter_scp.pl ${data_dir}/train_combined_no_sil/utt2num_frames ${data_dir}/train_combined_no_sil/utt2spk > ${data_dir}/train_combined_no_sil/utt2spk.new
  mv ${data_dir}/train_combined_no_sil/utt2spk.new ${data_dir}/train_combined_no_sil/utt2spk
  utils/fix_data_dir.sh ${data_dir}/train_combined_no_sil

  utils/filter_scp.pl ${data_dir}/train_combined_no_sil/utt2spk ${data_dir}/train_combined_no_sil/utt2num_frames > ${data_dir}/train_combined_no_sil/utt2num_frames.new
  mv ${data_dir}/train_combined_no_sil/utt2num_frames.new ${data_dir}/train_combined_no_sil/utt2num_frames

  # Now we're ready to create training examples.
  utils/fix_data_dir.sh ${data_dir}/train_combined_no_sil
fi

# Stages 8 and 9 (generating egs and training the nnet) are handled in run_meld_xvector.sh
if [ $stage -ge 8 ]; then
  if [ $stage -le 9 ]; then
    local/nnet3/xvector/run_meld_xvector.sh --stage $stage --train-stage -1 \
      --data ${data_dir}/train_combined_no_sil --nnet-dir $nnet_dir \
      --egs-dir $nnet_dir/egs --input-model "${MODEL_OUTPUT_DIR}/${MODIFIED_REFERENCE_MODEL}"
  fi
fi
