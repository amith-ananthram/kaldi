# author: aa4461

. ./cmd.sh
. ./path.sh

set -e

stage=placeholder
model_path=placeholder
target_emotions_mode=placeholder
target_emotions_config=placeholder
test_set=all_iemocap

. ./utils/parse_options.sh

BASE_DIR="nnet-emotion/evaluate"
MFCC_DIR="${BASE_DIR}/mfcc"

# prepare the corpus for feature extraction
if [ $stage -eq 0 ]; then
	echo "Stage $stage: start"
	for session in 1 2 3 4 5; do
		session_output_path="${BASE_DIR}/session${session}"
		mkdir -p $session_output_path
		nnet-emotion/detector/scoring/generate_iemocap_inputs.py \
			--test_corpus "iemocap${session}" \
			--test_corpus_config "all" \
			--target_emotions_mode $target_emotions_mode \
			--target_emotions_config $target_emotions_config \
			--output_dir $session_output_path
		utils/utt2spk_to_spk2utt.pl "${session_output_path}/utt2spk" > "${session_output_path}/spk2utt"
	done
	echo "Stage $stage: end"
fi

# extract MFCC and pitch features, combine data
if [ $stage -eq 1 ]; then
	echo "Stage $stage: start"
	for session in 1 2 3 4 5; do
		session_input_path="${BASE_DIR}/iemocap${session}" 
		steps/make_mfcc_pitch.sh --write-utt2num-frames true --mfcc-config conf/mfcc.conf --pitch-config conf/pitch.conf --nj 40 --cmd "$train_cmd" \
			$session_input_path ${BASE_DIR}/exp/make_mfcc $MFCC_DIR
		utils/fix_data_dir.sh $session_input_path
		sid/compute_vad_decision.sh --nj 40 --cmd "$train_cmd" \
      		$session_input_path ${BASE_DIR}/exp/make_vad $MFCC_DIR
      	utils/fix_data_dir.sh $session_input_path
	done
	utils/combine_data.sh "${BASE_DIR}/all_iemocap" "${BASE_DIR}/iemocap1" "${BASE_DIR}/iemocap2" "${BASE_DIR}/iemocap3" "${BASE_DIR}/iemocap4" "${BASE_DIR}/iemocap5"
	echo "Stage $stage: end"
fi

# run extracted features through the nnet
if [ $stage -eq 2 ]; then
	echo "Stage $stage: generating predictions for specified model"
	mkdir -p "${BASE_DIR}/predictions"
	nnet3-xvector-compute-batched --use-gpu=yes "${model_path}/final.raw" scp:${BASE_DIR}/$test_set/feats.scp ark:${BASE_DIR}/predictions/iemocap_predictions.ark
	echo "Stage $stage: end"
fi

# score nnet predictions against actual labels
if [ $stage -eq 3 ]; then 
	echo "Stage $stage: generating scores for specified model"
	mkdir -p "${BASE_DIR}/scores"
	nnet-emotion/detector/scoring/score_emotion_prediction_results.py \
		"${BASE_DIR}/$test_set/utt2spk" \
		"${BASE_DIR}/predictions/iemocap_predictions.ark" \
		"${BASE_DIR}/scores/iemocap_scores.txt"
	echo "Stage $stage: end"
fi
