. ./cmd.sh
. ./path.sh

set -e

stage=placeholder
model_path=placeholder
iemocap_path=placeholder
output_path=placeholder

. ./utils/parse_options.sh

BASE_DIR="${iemocap_path}/IEMOCAP_full_release"
CSVS_DIR="${BASE_DIR}/csvs"
DATA_DIR="${BASE_DIR}"

MFCC_DIR="${output_path}/mfcc"

# prepare the corpus for feature extraction
if [ $stage -eq 0 ]; then
	echo "Stage 0: start"
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
	echo "Stage 0: end"
fi

# extract MFCC and pitch features
if [ $stage -eq 1 ]; then
	echo "Stage 1: start"
	steps/make_mfcc_pitch.sh --write-utt2num-frames true --mfcc-config conf/mfcc.conf --pitch-config conf/pitch.conf --nj 40 --cmd "$train_cmd" \
		"${output_path}/all_iemocap" ${BASE_DIR}/exp/make_mfcc $MFCC_DIR
	utils/fix_data_dir.sh "${output_path}/all_iemocap"
        echo "Stage 1: end"
fi

# run extracted features through the nnet
if [ $stage -eq 2 ]; then
        echo "Stage 2: start"
	mkdir -p "${output_path}/predictions"
	for mode in no_sil with_sil; do
		for min_frame_len in 100 150 200 250 300; do 
			model="${mode}_${min_frame_len}"
			nnet3-compute-batch "${model_path}/${model}.raw" scp:${output_path}/all_iemocap/feats.scp ark:${output_path}/predictions/${model}_prediction.ark
        	done
	done
	echo "Stage 2: end"
fi

# score nnet predictions against actual labels
if [ $stage -eq 3 ]; then
	echo "Stage 3: start"
	mkdir -p "${output_path}/scores"
	for mode in no_sil with_sil; do
		for min_frame_len in 100 150 200 250 300; do
			model="${mode}_${min_frame_len}"
			score_emotion_prediction_results.py \
				$model \
				"${output_path}/all_iemocap/utt2spk" \
				"${output_path}/predictions/${model}_prediction.ark" \
				"${output_path}/scores"
		done
	done
        echo "Stage 3: end"
fi
