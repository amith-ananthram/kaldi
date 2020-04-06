. ./cmd.sh
. ./path.sh

set -e

model_path=placeholder
num_layers=placeholder
corpus_dir=placeholder
output_dir=placeholder

. ./utils/parse_options.sh

start_layer=4
min_chunk_size=25
max_chunk_size=10000

for layer in $(eval echo "{$start_layer..$num_layers}")
do
	echo "Extracting x-vectors from layer ${layer} of ${model_path}..."
	
	# first we configure the extraction (which layer) and then copy the source
	# model into a temporary working directory (to avoid corrupting our data)
	workdir="temp"
	mkdir $workdir
	echo "$min_chunk_size" > $workdir/min_chunk_size
	echo "$max_chunk_size" > $workdir/max_chunk_size
	echo "output-node name=output input=tdnn${layer}.affine" > $workdir/extract.config
	cp $model_path temp/final.raw

	outdir="$output_dir/$layer"
	mkdir $outdir
	sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 4G" --nj 40 \
		$work_dir $corpus_dir $outdir

	# clean up (so we hard fail if we run into any issues on the next iteration)
	rm -rf temp
done