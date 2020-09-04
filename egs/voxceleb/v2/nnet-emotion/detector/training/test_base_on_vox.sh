. ./cmd.sh
. ./path.sh
set -e

base_dir=nnet-emotion/vox_test
mfccdir=$base_dir/mfcc
vaddir=$base_dir/mfcc
datadir=$base_dir/data
nnetdir=$base_dir/nnetdir

voxceleb1_root=corpora/voxceleb/vox1
voxceleb2_root=corpora/voxceleb/vox2
voxceleb1_trials=$datadir/voxceleb1_test/trials

stage=0

. ./utils/parse_options.sh

# get utt2spk files for vox1 and vox2
if [ $stage -le 0 ]; then
	local/make_voxceleb1_v2.pl $voxceleb1_root dev $datadir/voxceleb1_train
	local/make_voxceleb1_v2.pl $voxceleb1_root test $datadir/voxceleb1_test
	local/make_voxceleb2.pl $voxceleb2_root dev $datadir/voxceleb2_train
	local/make_voxceleb2.pl $voxceleb2_root test $datadir/voxceleb2_test
	utils/combine_data.sh $datadir/train $datadir/voxceleb1_train $datadir/voxceleb2_train $datadir/voxceleb2_test
fi 

# calculate MFCCs and VAD (uses pitch config)
if [ $stage -le 1 ]; then
	for name in train voxceleb1_test; do
		steps/make_mfcc_pitch.sh --write-utt2num-frames true --mfcc-config conf/mfcc.conf --pitch-config conf/pitch.conf --nj 40 --cmd "$train_cmd" \
			$datadir/$name $base_dir/exp/make_mfcc $mfccdir
		utils/fix_data_dir.sh $datadir/$name
		sid/compute_vad_decision.sh --nj 40 --cmd "$train_cmd" \
			$datadir/$name $base_dir/exp/make_vad $vaddir
		utils/fix_data_dir.sh $datadir/$name
  	done
fi

# extract x-vectors
if [ $stage -le 2 ]; then
	mkdir -p $nnetdir
	cp vox2_base.raw $nnetdir/final.raw
	echo "25" > $nnetdir/min_chunk_size
	echo "10000" > $nnetdir/max_chunk_size
	echo "output-node name=output input=tdnn6.affine" > $nnetdir/extract.config
	sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 4G" --nj 80 \
		$nnetdir $datadir/train \
		$nnetdir/xvectors_train

	sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 4G" --nj 40 \
		$nnetdir $datadir/voxceleb1_test \
		$nnetdir/xvectors_voxceleb1_test
fi

# train LDA/pLDA
if [ $stage -le 3 ]; then
  # Compute the mean vector for centering the evaluation xvectors.
  $train_cmd $nnetdir/xvectors_train/log/compute_mean.log \
    ivector-mean scp:$nnetdir/xvectors_train/xvector.scp \
    $nnetdir/xvectors_train/mean.vec || exit 1;

  # This script uses LDA to decrease the dimensionality prior to PLDA.
  lda_dim=200
  $train_cmd $nnetdir/xvectors_train/log/lda.log \
    ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
    "ark:ivector-subtract-global-mean scp:$nnetdir/xvectors_train/xvector.scp ark:- |" \
    ark:$datadir/train/utt2spk $nnetdir/xvectors_train/transform.mat || exit 1;

  # Train the PLDA model.
  $train_cmd $nnetdir/xvectors_train/log/plda.log \
    ivector-compute-plda ark:$datadir/train/spk2utt \
    "ark:ivector-subtract-global-mean scp:$nnetdir/xvectors_train/xvector.scp ark:- | transform-vec $nnetdir/xvectors_train/transform.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
    $nnetdir/xvectors_train/plda || exit 1;
fi

# compute scores for test file
if [ $stage -le 4 ]; then
  $train_cmd $base_dir/exp/scores/log/voxceleb1_test_scoring.log \
    ivector-plda-scoring --normalize-length=true \
    "ivector-copy-plda --smoothing=0.0 $nnetdir/xvectors_train/plda - |" \
    "ark:ivector-subtract-global-mean $nnetdir/xvectors_train/mean.vec scp:$nnetdir/xvectors_voxceleb1_test/xvector.scp ark:- | transform-vec $nnetdir/xvectors_train/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean $nnetdir/xvectors_train/mean.vec scp:$nnetdir/xvectors_voxceleb1_test/xvector.scp ark:- | transform-vec $nnetdir/xvectors_train/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$voxceleb1_trials' | cut -d\  --fields=1,2 |" $base_dir/exp/scores_voxceleb1_test || exit 1;
fi

# compute EER
if [ $stage -le 5 ]; then
  eer=`compute-eer <(local/prepare_for_eer.py $voxceleb1_trials $base_dir/exp/scores_voxceleb1_test) 2> /dev/null`
  mindcf1=`sid/compute_min_dcf.py --p-target 0.01 $base_dir/exp/scores_voxceleb1_test $voxceleb1_trials 2> /dev/null`
  mindcf2=`sid/compute_min_dcf.py --p-target 0.001 $base_dir/exp/scores_voxceleb1_test $voxceleb1_trials 2> /dev/null`
  echo "EER: $eer%"
  echo "minDCF(p-target=0.01): $mindcf1"
  echo "minDCF(p-target=0.001): $mindcf2"
fi
