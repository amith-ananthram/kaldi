. ./cmd.sh
. ./path.sh

set -e

stage=placeholder
model_path=placeholder
meld_path=placeholder
output_path=placeholder

. ./utils/parse_options.sh

BASE_DIR="${meld_path}"

# run extracted features through the nnet
if [ $stage -eq 0 ]; then
	echo "Stage 0: start"
	mkdir -p "${output_path}/predictions"
	for layers in seven_layers eight_layers; do
		for mode in no_sil with_sil; do
			for min_frame_len in 100 150 200 250 300; do 
				model="${layers}_${mode}_${min_frame_len}"
				if [ ! -f "${model_path}/${model}.raw" ]
				then
					echo "${model} does not exist!"
					continue
				fi
				if [ -f "${output_path}/predictions/${model}_prediction.ark" ]
				then
					echo "${model} predictions exist, skipping"
					continue
				fi
				nnet3-compute-batch "${model_path}/${model}.raw" scp:${output_path}/outputs/data/all_meld/feats.scp ark:${output_path}/predictions/${model}_prediction.ark
			done
		done
	done
	echo "Stage 0: end"
fi

# score nnet predictions against actual labels
if [ $stage -eq 1 ]; then
	echo "Stage 1: start"
	mkdir -p "${output_path}/scores"
	for layers in seven_layers eight_layers; do
		for mode in no_sil with_sil; do
			for min_frame_len in 100 150 200 250 300; do
				model="${layers}_${mode}_${min_frame_len}"
				score_emotion_prediction_results.py \
					$model \
					"${output_path}/outputs/data/all_meld/utt2spk" \
					"${output_path}/predictions/${model}_prediction.ark" \
					"${output_path}/scores"
			done
		done
	done
	echo "Stage 1: end"
fi