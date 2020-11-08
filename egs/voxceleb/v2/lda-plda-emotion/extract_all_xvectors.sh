set -e 

# grouped variants!

# extract training x-vectors

lda-plda-emotion/extract_xvectors.sh \
	--nj 40 \
	--use_gpu false \
	--model_path models/3_1/final.raw \
	--num_layers 7 \
	--corpus_dir nnet-emotion/finetune/outputs/data/combined \
	--output_base_dir models/3_1/xvectors \
	--output_dir cremad

# extract evaluation x-vectors

lda-plda-emotion/extract_xvectors.sh \
	--nj 40 \
	--use_gpu false \
	--model_path models/3_1/final.raw \
	--num_layers 7 \
	--corpus_dir nnet-emotion/evaluate/all_iemocap/ \
	--output_base_dir models/3_1/xvectors \
	--output_dir iemocap

# extract training x-vectors

lda-plda-emotion/extract_xvectors.sh \
	--nj 40 \
	--use_gpu false \
	--model_path models/3_2/final.raw \
	--num_layers 7 \
	--corpus_dir nnet-emotion/finetune/outputs/data/combined \
	--output_base_dir models/3_2/xvectors \
	--output_dir cremad

# extract evaluation x-vectors

lda-plda-emotion/extract_xvectors.sh \
	--nj 40 \
	--use_gpu false \
	--model_path models/3_2/final.raw \
	--num_layers 7 \
	--corpus_dir nnet-emotion/evaluate/all_iemocap/ \
	--output_base_dir models/3_2/xvectors \
	--output_dir iemocap

# extract training x-vectors

lda-plda-emotion/extract_xvectors.sh \
	--nj 40 \
	--use_gpu false \
	--model_path models/3_3/final.raw \
	--num_layers 8 \
	--corpus_dir nnet-emotion/finetune/outputs/data/combined \
	--output_base_dir models/3_3/xvectors \
	--output_dir cremad

# extract evaluation x-vectors

lda-plda-emotion/extract_xvectors.sh \
	--nj 40 \
	--use_gpu false \
	--model_path models/3_3/final.raw \
	--num_layers 8 \
	--corpus_dir nnet-emotion/evaluate/all_iemocap/ \
	--output_base_dir models/3_3/xvectors \
	--output_dir iemocap

# todo: ungrouped variants!