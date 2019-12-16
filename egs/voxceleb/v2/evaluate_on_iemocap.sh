. ./cmd.sh
. ./path.sh

set -e

stage=placeholder
model_path=placeholder
iemocap_path=placeholder
output_path=placeholder
experiment_name=placeholder

. ./utils/parse_options.sh

BASE_DIR="${iemocap_path}/IEMOCAP_full_release"
CSVS_DIR="${BASE_DIR}/csvs"
DATA_DIR="${BASE_DIR}"

MFCC_DIR="${output_path}/mfcc"

# prepare the corpus for feature extraction
if [ $stage -eq 0 ]; then
	for session in 1 2 3 4 5; do
		session_output_path="${output_path}/session${session}"
		mkdir -p $session_output_path
		generate_iemocap_inputs.py \
			"${CSVS_DIR}/Session${session}.csv" \
			"${DATA_DIR}/Session${session}/sentences/wav" \
			$session_output_path
		utils/utt2spk_to_spk2utt.pl "${session_output_path}/utt2spk" > "${session_output_path}/spk2utt"
	done
	utils/combine_data.sh "${output_path}/all_iemocap" "${output_path}/session1" "${output_path}/session2" "${output_path}/session3" "${output_path}/session4" "${output_path}/session5"
fi

# extract MFCC and pitch features
if [ $stage -eq 1 ]; then
	steps/make_mfcc_pitch.sh --write-utt2num-frames true --mfcc-config conf/mfcc.conf --pitch-config conf/pitch.conf --nj 40 --cmd "$train_cmd" \
		"${output_path}/all_iemocap" ${BASE_DIR}/exp/make_mfcc $MFCC_DIR
	utils/fix_data_dir.sh "${output_path}/all_iemocap"
fi

# run extracted features through the nnet
if [ $stage -eq 2 ]; then
	nnet3-compute $model_path scp:${output_path}/all_iemocap/feats.scp ark:${output_path}/${experiment_name}_prediction.ark
fi

# score nnet predictions against actual labels
if [ $stage -eq 3 ]; then
	score_emotion_prediction_results.py \
		$experiment_name \
		"${output_path}/all_iemocap/utt2spk" \
		${output_path}/${experiment_name}_prediction.ark
fi