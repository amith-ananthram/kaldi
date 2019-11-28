#!/bin/bash
# Copyright      2017   David Snyder
#                2017   Johns Hopkins University (Author: Daniel Garcia-Romero)
#                2017   Johns Hopkins University (Author: Daniel Povey)
#
# Copied from egs/sre16/v1/local/nnet3/xvector/tuning/run_xvector_1a.sh (commit e082c17d4a8f8a791428ae4d9f7ceb776aef3f0b).
#
# Apache 2.0.

# This script trains a DNN similar to the recipe described in
# http://www.danielpovey.com/files/2018_icassp_xvectors.pdf

. ./cmd.sh
set -e

stage=1
train_stage=0
use_gpu=true
remove_egs=false

data=placeholder
nnet_dir=placeholder
egs_dir=placeholder
input_model=placeholder

. ./path.sh
. ./cmd.sh
. ./utils/parse_options.sh

# Now we create the nnet examples using sid/nnet3/xvector/get_egs.sh.
# The argument --num-repeats is related to the number of times a speaker
# repeats per archive.  If it seems like you're getting too many archives
# (e.g., more than 200) try increasing the --frames-per-iter option.  The
# arguments --min-frames-per-chunk and --max-frames-per-chunk specify the
# minimum and maximum length (in terms of number of frames) of the features
# in the examples.
#
# To make sense of the egs script, it may be necessary to put an "exit 1"
# command immediately after stage 3.  Then, inspect
# exp/<your-dir>/egs/temp/ranges.* . The ranges files specify the examples that
# will be created, and which archives they will be stored in.  Each line of
# ranges.* has the following form:
#    <utt-id> <local-ark-indx> <global-ark-indx> <start-frame> <end-frame> <spk-id>
# For example:
#    100304-f-sre2006-kacg-A 1 2 4079 881 23

# If you're satisfied with the number of archives (e.g., 50-150 archives is
# reasonable) and with the number of examples per speaker (e.g., 1000-5000
# is reasonable) then you can let the script continue to the later stages.
# Otherwise, try increasing or decreasing the --num-repeats option.  You might
# need to fiddle with --frames-per-iter.  Increasing this value decreases the
# the number of archives and increases the number of examples per archive.
# Decreasing this value increases the number of archives, while decreasing the
# number of examples per archive.
if [ $stage -eq 8 ]; then
  echo "$0: Getting neural network training egs";
  # dump egs.
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $egs_dir/storage ]; then
    utils/create_split_dir.pl \
     /export/b{03,04,05,06}/$USER/kaldi-data/egs/voxceleb2/v2/xvector-$(date +'%m_%d_%H_%M')/$egs_dir/storage $egs_dir/storage
  fi
  sid/nnet3/xvector/get_egs.sh --cmd "$train_cmd" \
    --nj 8 \
    --stage 0 \
    --frames-per-iter 1000000000 \
    --frames-per-iter-diagnostic 100000 \
    --min-frames-per-chunk 50 \
    --max-frames-per-chunk 100 \
    --num-diagnostic-archives 3 \
    --num-repeats 50 \
    "$data" $egs_dir
fi

dropout_schedule='0,0@0.20,0.1@0.50,0'
srand=123
if [ $stage -eq 9 ]; then
  steps/nnet3/train_raw_dnn.py --stage=$train_stage \
    --cmd="$train_cmd" \
    --trainer.input-model "$input_model" \
    --trainer.optimization.proportional-shrink 10 \
    --trainer.optimization.momentum=0.5 \
    --trainer.optimization.num-jobs-initial=1 \
    --trainer.optimization.num-jobs-final=3 \
    --trainer.optimization.initial-effective-lrate=0.001 \
    --trainer.optimization.final-effective-lrate=0.0001 \
    --trainer.optimization.minibatch-size=64 \
    --trainer.srand=$srand \
    --trainer.max-param-change=2 \
    --trainer.num-epochs=3 \
    --trainer.dropout-schedule="$dropout_schedule" \
    --trainer.shuffle-buffer-size=1000 \
    --egs.frames-per-eg=1 \
    --egs.dir="$egs_dir" \
    --cleanup.remove-egs $remove_egs \
    --cleanup.preserve-model-interval=10 \
    --use-gpu=true \
    --dir=$nnet_dir  || exit 1;
fi

exit 0;
