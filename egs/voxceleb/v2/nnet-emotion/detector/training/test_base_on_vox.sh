. ./cmd.sh
. ./path.sh
set -e

base_dir=nnet-emotion/vox_test
mfccdir=$base_dir/mfcc
vaddir=$base_dir/mfcc
datadir=$base_dir/data

voxceleb1_root=corpora/voxceleb/vox1
voxceleb2_root=corpora/voxceleb/vox2

stage=0

. ./utils/parse_options.sh

# get utt2spk files for vox1 and vox2
if [ $stage -eq 0 ]; then
	local/make_voxceleb1_v2.pl $voxceleb1_root test $datadir/voxceleb1_test
	local/make_voxceleb2.pl $voxceleb2_root test $datadir/voxceleb2_test
	utils/combine_data.sh $datadir/evaluate $datadir/voxceleb1_test $datadir/voxceleb2_test
fi 

# calculate MFCCs and VAD (uses pitch config)
if [ $stage -eq 1 ]; then
  steps/make_mfcc.sh --write-utt2num-frames true --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" \
    $datadir/evaluate nnet-emotion/vox-test/exp/make_mfcc $mfccdir
  utils/fix_data_dir.sh $datadir/evaluate
  sid/compute_vad_decision.sh --nj 40 --cmd "$train_cmd" \
    $datadir/evaluate nnet-emotion/vox-test/exp/make_vad $vaddir
  utils/fix_data_dir.sh $datadir/evaluate
fi

# run extracted features through nnet 
if [ $stage -eq 2 ]; then
	echo "Stage $stage: generating predictions for specified model"
	mkdir -p "${base_dir}/predictions"
	nnet3-xvector-compute-batched --use-gpu=yes vox2_base.raw scp:$datadir/evaluate/feats.scp ark:${base_dir}/predictions/vox_predictions.ark
	echo "Stage $stage: end"
fi
