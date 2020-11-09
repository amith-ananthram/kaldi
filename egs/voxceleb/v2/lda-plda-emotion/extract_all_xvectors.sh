set -e 

# 3_1

lda-plda-emotion/extract_xvectors.sh \
	--nj 40 \
	--use_gpu false \
	--model_path models/3_1/final.raw \
	--num_layers 7 \
	--corpus_dir nnet-emotion/finetune/outputs/data/combined \
	--output_base_dir models/3_1/xvectors \
	--output_dir cremad

lda-plda-emotion/extract_xvectors.sh \
	--nj 40 \
	--use_gpu false \
	--model_path models/3_1/final.raw \
	--num_layers 7 \
	--corpus_dir nnet-emotion/evaluate/all_iemocap/ \
	--output_base_dir models/3_1/xvectors \
	--output_dir iemocap

# 3_2

lda-plda-emotion/extract_xvectors.sh \
	--nj 40 \
	--use_gpu false \
	--model_path models/3_2/final.raw \
	--num_layers 7 \
	--corpus_dir nnet-emotion/finetune/outputs/data/combined \
	--output_base_dir models/3_2/xvectors \
	--output_dir cremad

lda-plda-emotion/extract_xvectors.sh \
	--nj 40 \
	--use_gpu false \
	--model_path models/3_2/final.raw \
	--num_layers 7 \
	--corpus_dir nnet-emotion/evaluate/all_iemocap/ \
	--output_base_dir models/3_2/xvectors \
	--output_dir iemocap

# 3_3

lda-plda-emotion/extract_xvectors.sh \
	--nj 40 \
	--use_gpu false \
	--model_path models/3_3/final.raw \
	--num_layers 8 \
	--corpus_dir nnet-emotion/finetune/outputs/data/combined \
	--output_base_dir models/3_3/xvectors \
	--output_dir cremad

lda-plda-emotion/extract_xvectors.sh \
	--nj 40 \
	--use_gpu false \
	--model_path models/3_3/final.raw \
	--num_layers 8 \
	--corpus_dir nnet-emotion/evaluate/all_iemocap/ \
	--output_base_dir models/3_3/xvectors \
	--output_dir iemocap

# 3_4

lda-plda-emotion/extract_xvectors.sh \
	--nj 40 \
	--use_gpu false \
	--model_path models/3_4/final.raw \
	--num_layers 7 \
	--corpus_dir nnet-emotion/finetune/outputs/data/combined \
	--output_base_dir models/3_4/xvectors \
	--output_dir cremad

lda-plda-emotion/extract_xvectors.sh \
	--nj 40 \
	--use_gpu false \
	--model_path models/3_4/final.raw \
	--num_layers 7 \
	--corpus_dir nnet-emotion/evaluate/all_iemocap/ \
	--output_base_dir models/3_4/xvectors \
	--output_dir iemocap

# 3_5

lda-plda-emotion/extract_xvectors.sh \
	--nj 40 \
	--use_gpu false \
	--model_path models/3_5/final.raw \
	--num_layers 7 \
	--corpus_dir nnet-emotion/finetune/outputs/data/combined \
	--output_base_dir models/3_5/xvectors \
	--output_dir cremad

lda-plda-emotion/extract_xvectors.sh \
	--nj 40 \
	--use_gpu false \
	--model_path models/3_5/final.raw \
	--num_layers 7 \
	--corpus_dir nnet-emotion/evaluate/all_iemocap/ \
	--output_base_dir models/3_5/xvectors \
	--output_dir iemocap

# 3_6

lda-plda-emotion/extract_xvectors.sh \
	--nj 40 \
	--use_gpu false \
	--model_path models/3_6/final.raw \
	--num_layers 8 \
	--corpus_dir nnet-emotion/finetune/outputs/data/combined \
	--output_base_dir models/3_6/xvectors \
	--output_dir cremad

lda-plda-emotion/extract_xvectors.sh \
	--nj 40 \
	--use_gpu false \
	--model_path models/3_6/final.raw \
	--num_layers 8 \
	--corpus_dir nnet-emotion/evaluate/all_iemocap/ \
	--output_base_dir models/3_6/xvectors \
	--output_dir iemocap

# todo: ungrouped variants for iemocap emotion confirmation evaluation!