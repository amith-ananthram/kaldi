# model 16, layer 6, speech only
lda-plda-emotion/lda_plda.sh \
	--variant "00000" \
	--speech_dir models/cremad/16/xvectors/6/ \
	--text_dir none \
	--train_corpora cremad \
	--output_dir lda_output
lda-plda-emotion/lda_plda.sh \
	--variant "00001" \
	--speech_dir models/cremad/16/xvectors/6/ \
	--text_dir none \
	--train_corpora cremad,iemocap1 \
	--output_dir lda_output
lda-plda-emotion/lda_plda.sh \
	--variant "00002" \
	--speech_dir models/cremad/16/xvectors/6/ \
	--text_dir none \
	--train_corpora cremad,iemocap2 \
	--output_dir lda_output
lda-plda-emotion/lda_plda.sh \
	--variant "00003" \
	--speech_dir models/cremad/16/xvectors/6/ \
	--text_dir none \
	--train_corpora cremad,iemocap3 \
	--output_dir lda_output
lda-plda-emotion/lda_plda.sh \
	--variant "00004" \
	--speech_dir models/cremad/16/xvectors/6/ \
	--text_dir none \
	--train_corpora cremad,iemocap4 \
	--output_dir lda_output
lda-plda-emotion/lda_plda.sh \
	--variant "00005" \
	--speech_dir models/cremad/16/xvectors/6/ \
	--text_dir none \
	--train_corpora cremad,iemocap5 \
	--output_dir lda_output
# model 16, layer 7, speech only
lda-plda-emotion/lda_plda.sh \
	--variant "00006" \
	--speech_dir models/cremad/16/xvectors/7/ \
	--text_dir none \
	--train_corpora cremad \
	--output_dir lda_output
lda-plda-emotion/lda_plda.sh \
	--variant "00007" \
	--speech_dir models/cremad/16/xvectors/7/ \
	--text_dir none \
	--train_corpora cremad,iemocap1 \
	--output_dir lda_output
lda-plda-emotion/lda_plda.sh \
	--variant "00008" \
	--speech_dir models/cremad/16/xvectors/7/ \
	--text_dir none \
	--train_corpora cremad,iemocap2 \
	--output_dir lda_output
lda-plda-emotion/lda_plda.sh \
	--variant "00009" \
	--speech_dir models/cremad/16/xvectors/7/ \
	--text_dir none \
	--train_corpora cremad,iemocap3 \
	--output_dir lda_output
lda-plda-emotion/lda_plda.sh \
	--variant "00010" \
	--speech_dir models/cremad/16/xvectors/7/ \
	--text_dir none \
	--train_corpora cremad,iemocap4 \
	--output_dir lda_output
lda-plda-emotion/lda_plda.sh \
	--variant "00011" \
	--speech_dir models/cremad/16/xvectors/7/ \
	--text_dir none \
	--train_corpora cremad,iemocap5 \
	--output_dir lda_output