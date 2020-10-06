#!/bin/bash

set -e -o pipefail


# This script is called from local/nnet3/run_tdnn.sh and local/chain/run_tdnn.sh (and may eventually
# be called by more scripts).  It contains the common feature preparation and iVector-related parts
# of the script.  See those scripts for examples of usage.


stage=0
nj=30

train_set=train_cleaned   # you might set this to e.g. train.
gmm=tri3_cleaned          # This specifies a GMM-dir from the features of the type you're training the system on;
                          # it should contain alignments for 'train_set'.
online_cmvn_iextractor=false

num_threads_ubm=8
nnet3_affix=_cleaned     # affix for exp/nnet3 directory to put iVector stuff in, so it
                         # becomes exp/nnet3_cleaned or whatever.

xvector_period=10
xvector_nnet_dir=placeholder

. ./cmd.sh
. ./path.sh
. utils/parse_options.sh


gmm_dir=exp/${gmm}
ali_dir=exp/${gmm}_ali_${train_set}_sp

for f in data/${train_set}/feats.scp ${gmm_dir}/final.mdl; do
  if [ ! -f $f ]; then
    echo "$0: expected file $f to exist"
    exit 1
  fi
done


# lowres features, alignments
if [ -f data/${train_set}_sp/feats.scp ] && [ $stage -le 2 ]; then
  echo "$0: data/${train_set}_sp/feats.scp already exists.  Refusing to overwrite the features "
  echo " to avoid wasting time.  Please remove the file and continue if you really mean this."
  exit 1;
fi

if [ $stage -le 1 ]; then
  echo "$0: preparing directory for low-resolution speed-perturbed data (for alignment)"
  utils/data/perturb_data_dir_speed_3way.sh \
    data/${train_set} data/${train_set}_sp

  for datadir in ${train_set}_sp dev test; do
    utils/copy_data_dir.sh data/$datadir data/${datadir}_hires
  done
fi

if [ $stage -le 2 ]; then
  echo "$0: making MFCC features for low-resolution speed-perturbed data"
  steps/make_mfcc_pitch.sh --nj $nj \
    --cmd "$train_cmd" data/${train_set}_sp
  steps/compute_cmvn_stats.sh data/${train_set}_sp
  echo "$0: fixing input data-dir to remove nonexistent features, in case some "
  echo ".. speed-perturbed segments were too short."
  utils/fix_data_dir.sh data/${train_set}_sp
fi

if [ $stage -le 3 ]; then
  if [ -f $ali_dir/ali.1.gz ]; then
    echo "$0: alignments in $ali_dir appear to already exist.  Please either remove them "
    echo " ... or use a later --stage option."
    exit 1
  fi
  echo "$0: aligning with the perturbed low-resolution data"
  steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
         data/${train_set}_sp data/lang $gmm_dir $ali_dir
fi


if [ $stage -le 5 ] && [ -f data/${train_set}_sp_hires/feats.scp ]; then
  echo "$0: data/${train_set}_sp_hires/feats.scp already exists."
  echo " ... Please either remove it, or rerun this script with stage > 2."
  exit 1
fi

if [ $stage -le 5 ]; then
  echo "$0: creating high-resolution MFCC features (for TDNN training)"  

  # this shows how you can split across multiple file-systems.  we'll split the
  # MFCC dir across multiple locations.  You might want to be careful here, if you
  # have multiple copies of Kaldi checked out and run the same recipe, not to let
  # them overwrite each other.
  mfccdir=data/${train_set}_sp_hires/data
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $mfccdir/storage ]; then
    utils/create_split_dir.pl /export/b0{5,6,7,8}/$USER/kaldi-data/mfcc/tedlium-$(date +'%m_%d_%H_%M')/s5/$mfccdir/storage $mfccdir/storage
  fi

  # do volume-perturbation on the training data prior to extracting hires
  # features; this helps make trained nnets more invariant to test data volume.
  utils/data/perturb_data_dir_volume.sh data/${train_set}_sp_hires

  for datadir in ${train_set}_sp dev test; do
    steps/make_mfcc_pitch.sh --nj $nj --mfcc-config conf/mfcc_hires.conf \
      --cmd "$train_cmd" data/${datadir}_hires
    steps/compute_cmvn_stats.sh data/${datadir}_hires
    utils/fix_data_dir.sh data/${datadir}_hires
  done
fi

if [ $stage -le 6 ]; then 
  echo "$0: creating low-resolution MFCC features for x-vector extraction"

  for datadir in ${train_set}_sp dev test; do
    utils/copy_data_dir.sh data/$datadir data/${datadir}_lores
  done

  # do volume-perturbation on the training data prior to extracting lores
  # features; this helps make trained nnets more invariant to test data volume.
  utils/data/perturb_data_dir_volume.sh data/${train_set}_sp_lores

  for datadir in ${train_set}_sp dev test; do
    steps/make_mfcc_pitch.sh --nj $nj --mfcc-config conf/mfcc_lores.conf \
      --cmd "$train_cmd" data/${datadir}_lores data/${datadir}_lores/log data/${datadir}_lores/mfcc
    steps/compute_cmvn_stats.sh data/${datadir}_lores
    compute_vad_decision.sh --nj 40 --cmd "$train_cmd" \
      data/${datadir}_lores data/${datadir}_lores/log/make_vad data/${datadir}_lores/mfcc
    utils/fix_data_dir.sh data/${datadir}_lores
  done
fi

if [ $stage -le 7 ]; then
  echo "$0: extracting x-vectors from low-res MFCC features"

  for f in $xvector_nnet_dir/final.raw $xvector_nnet_dir/min_chunk_size $xvector_nnet_dir/max_chunk_size $xvector_nnet_dir/extract.config; do
    [ ! -f $f ] && echo "No such file $f" && exit 1;
  done

  for datadir in ${train_set}_sp; do # dev test
    if [ $datadir = ${train_set}_sp ]; then
      nj=80
      num_subsets=100
    else
      nj=30
      num_subsets=1
    fi

    #extract_xvectors.sh --cmd "$train_cmd --mem 4G" --nj 30 --chunk_size $xvector_period \
    #  $xvector_nnet_dir data/${datadir}_lores data/xvectors/${datadir}_lores

    python3 split_matrix_into_vectors.py \
      --src_xvector_scp "data/xvectors/${datadir}_lores/xvector.scp" \
      --src_xvector_utt2spk "data/${datadir}_lores/utt2spk" \
      --tgt_xvector_ark "data/xvectors/${datadir}_lores/xvector-split" \
      --tgt_xvector_scp "data/xvectors/${datadir}_lores/xvector-split" \
      --tgt_xvector_utt2spk "data/${datadir}_lores/utt2spk-split" \
      --num_subsets $num_subsets

    if [ $num_subsets -gt 1 ]; then
      for s in $(seq $num_subsets); do cat data/xvectors/${datadir}_lores/xvector-split.$s.scp; done >data/xvectors/${datadir}_lores/xvector-split.scp || exit 1;
      for s in $(seq $num_subsets); do cat data/xvectors/${datadir}_lores/utt2spk-split.$s; done >data/xvectors/${datadir}_lores/utt2spk-split || exit 1;
    done
  done
fi 

if [ $stage -le 8 ]; then
  echo "$0: training LDA to reduce x-vector dimensionality to 100" 

  # Compute the mean vector for centering the evaluation xvectors.
  $train_cmd data/xvectors/log/compute_mean.log \
    ivector-mean scp:data/xvectors/${train_set}_sp_lores/xvector-split.scp \
    data/xvectors/mean.vec || exit 1;

  # Trains LDA based off the xvectors
  lda_dim=100
  $train_cmd data/xvectors/log/lda.log \
    ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
    "ark:ivector-subtract-global-mean scp:data/xvectors/${train_set}_sp_lores/xvector-split.scp ark:- |" \
    ark:data/${train_set}_sp_lores/utt2spk-split data/xvectors/lda.mat || exit 1;
fi

if [ $stage -le 9 ]; then 
  echo "$0: reducing dimensionality of x-vectors"

  for datadir in dev test ${train_set}_sp; do
    ivector_dir=exp/nnet3${nnet3_affix}/ivectors_${datadir}_hires
    $train_cmd data/xvectors/log/reduce_${datadir}.log \
      ivector-normalize-length \
        "ark:ivector-subtract-global-mean data/xvectors/mean.vec scp:data/xvectors/${train_set}_sp_lores/xvector-split.scp ark:- | transform-vec data/xvectors/lda.mat ark:- ark:- |" \
        ark,scp:$ivector_dir/ivector_online-split.ark,$ivector_dir/ivector_online-split.scp

    echo $xvector_period > $ivector_dir/ivector_period

    python3 merge_vectors_into_matrix.py \
      --src_xvector_scp "$ivector_dir/ivector_online-split.scp" \
      --tgt_xvector_ark "$ivector_dir/ivector_online.ark" \
      --tgt_xvector_scp "$ivector_dir/ivector_online.scp"
  done
fi

exit 0;
