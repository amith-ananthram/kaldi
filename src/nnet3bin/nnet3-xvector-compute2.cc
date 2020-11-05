// nnet3bin/nnet3-xvector-compute2.cc

// Copyright 2017   Johns Hopkins University (author: Daniel Povey)
//           2017   Johns Hopkins University (author: Daniel Garcia-Romero)
//           2017   David Snyder

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.


#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "nnet3/nnet-am-decodable-simple.h"
#include "base/timer.h"
#include "nnet3/nnet-utils.h"

namespace kaldi {
namespace nnet3 {

// Computes an xvector from a chunk of speech features.
static void RunNnetComputation(const MatrixBase<BaseFloat> &features,
    const Nnet &nnet, CachingOptimizingCompiler *compiler,
    Vector<BaseFloat> *xvector) {
  ComputationRequest request;
  request.need_model_derivative = false;
  request.store_component_stats = false;
  request.inputs.push_back(
    IoSpecification("input", 0, features.NumRows()));
  IoSpecification output_spec;
  output_spec.name = "output";
  output_spec.has_deriv = false;
  output_spec.indexes.resize(1);
  request.outputs.resize(1);
  request.outputs[0].Swap(&output_spec);
  std::shared_ptr<const NnetComputation> computation(compiler->Compile(request));
  Nnet *nnet_to_update = NULL;  // we're not doing any update.
  NnetComputer computer(NnetComputeOptions(), *computation,
                  nnet, nnet_to_update);
  CuMatrix<BaseFloat> input_feats_cu(features);
  computer.AcceptInput("input", &input_feats_cu);
  computer.Run();
  CuMatrix<BaseFloat> cu_output;
  computer.GetOutputDestructive("output", &cu_output);
  xvector->Resize(cu_output.NumCols());
  xvector->CopyFromVec(cu_output.Row(0));
}

} // namespace nnet3
} // namespace kaldi

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Propagate features through an xvector neural network model and write\n"
        "the output vectors.  \"Xvector\" is our term for a vector or\n"
        "embedding which is the output of a particular type of neural network\n"
        "architecture found in speaker recognition.  This architecture\n"
        "consists of several layers that operate on frames, a statistics\n"
        "pooling layer that aggregates over the frame-level representations\n"
        "and possibly additional layers that operate on segment-level\n"
        "representations.  The xvectors are generally extracted from an\n"
        "output layer after the statistics pooling layer.  By default, one\n"
        "xvector is extracted directly from the set of features for each\n"
        "utterance.  Optionally, xvectors are extracted from chunks of input\n"
        "features and averaged, to produce a single vector.\n"
        "\n"
        "Usage: nnet3-xvector-compute2 [options] <raw-nnet-in> "
        "<features-rspecifier> <vector-wspecifier>\n"
        "e.g.: nnet3-xvector-compute2 final.raw scp:feats.scp "
        "ark:nnet_prediction.ark\n"
        "See also: nnet3-compute\n";

    ParseOptions po(usage);
    Timer timer;

    NnetSimpleComputationOptions opts;
    CachingOptimizingCompilerOptions compiler_config;

    opts.acoustic_scale = 1.0; // by default do no scaling in this recipe.

    std::string use_gpu = "no";
    std::string cached_compiler_in;
    std::string cached_compiler_out;
    int32 chunk_size = -1,
      min_chunk_size = 100;
    bool pad_input = true;

    opts.Register(&po);
    compiler_config.Register(&po);

    po.Register("use-gpu", &use_gpu,
      "yes|no|optional|wait, only has effect if compiled with CUDA");
    po.Register("chunk-size", &chunk_size,
      "If set, extracts xectors from specified chunk-size, and averages.  "
      "If not set, extracts an xvector from all available features.");
    po.Register("min-chunk-size", &min_chunk_size,
      "Minimum chunk-size allowed when extracting xvectors.");
    po.Register("pad-input", &pad_input, "If true, duplicate the first and "
      "last frames of the input features as required to equal min-chunk-size.");
    po.Register("cached-compiler-in", &cached_compiler_in,
      "If set, read the cached compiler from the specified file path.");
    po.Register("cached-compiler-out", &cached_compiler_out,
      "If set, write the cached compiler to the specified file path.");

#if HAVE_CUDA==1
    CuDevice::RegisterDeviceOptions(&po);
#endif

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

#if HAVE_CUDA==1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif

    std::string nnet_rxfilename = po.GetArg(1),
                feature_rspecifier = po.GetArg(2),
                vector_wspecifier = po.GetArg(3);

    Nnet nnet;
    ReadKaldiObject(nnet_rxfilename, &nnet);
    SetBatchnormTestMode(true, &nnet);
    SetDropoutTestMode(true, &nnet);
    CollapseModel(CollapseModelConfig(), &nnet);

    CachingOptimizingCompiler compiler(nnet, opts.optimize_config, compiler_config);
    
    if (!cached_compiler_in.empty()) {
        KALDI_LOG << "Reading cache from " << cached_compiler_in;
        bool cache_binary_in;
        Input ki(cached_compiler_in, &cache_binary_in);
        compiler.ReadCache(ki.Stream(), cache_binary_in);
    }

    BaseFloatMatrixWriter vector_writer(vector_wspecifier);

    int32 num_success = 0, num_fail = 0;
    int64 frame_count = 0;
    int32 xvector_dim = nnet.OutputDim("output");

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);

    for (; !feature_reader.Done(); feature_reader.Next()) {
      std::string utt = feature_reader.Key();
      const Matrix<BaseFloat> &features (feature_reader.Value());
      if (features.NumRows() == 0) {
        KALDI_WARN << "Zero-length utterance: " << utt;
        num_fail++;
        continue;
      }
      int32 num_rows = features.NumRows(),
            feat_dim = features.NumCols();

      int32 num_chunks = ceil(
        num_rows / static_cast<BaseFloat>(chunk_size));
      Matrix<BaseFloat> xvectors(num_chunks, xvector_dim);

      // Iterate over the feature chunks.  The chunk_indx indexes the center 
      // of our extraction window for our x-vector (which, if needed, is 
      // zero padded on the left and the right)
      for (int32 chunk_indx = 0; chunk_indx < num_chunks; chunk_indx++) {
        int32 chunk_center = chunk_indx * chunk_size;
        int32 chunk_start = chunk_center - (chunk_size - 1) / 2;
        int32 chunk_end = chunk_center + (chunk_size - 1) / 2;

        Vector<BaseFloat> xvector;
        Matrix<BaseFloat> sub_features(chunk_size, feat_dim);
        for (int32 i = chunk_start; i <= chunk_end; i++) {
          if (i < 0) {
            sub_features.Row(i - chunk_start).CopyFromVec(features.Row(0));
          else if (i >= num_rows) {
            sub_features.Row(i - chunk_start).CopyFromVec(features.Row(num_rows - 1));
          } else {
            sub_features.Row(i - chunk_start).CopyFromVec(features.Row(i));
          }
        }
        RunNnetComputation(sub_features, nnet, &compiler, &xvector);

        xvectors.CopyRowFromVec(xvector, chunk_indx);
      }
      vector_writer.Write(utt, xvectors);

      frame_count += features.NumRows();
      num_success++;
    }

#if HAVE_CUDA==1
    CuDevice::Instantiate().PrintProfile();
#endif
    double elapsed = timer.Elapsed();
    KALDI_LOG << "Time taken "<< elapsed
              << "s: real-time factor assuming 100 frames/sec is "
              << (elapsed*100.0/frame_count);
    KALDI_LOG << "Done " << num_success << " utterances, failed for "
              << num_fail;
    
    if (!cached_compiler_out.empty()) {
        KALDI_LOG << "Writing cache to " << cached_compiler_out;
        bool binary_write = true;
        Output ko(cached_compiler_out, &binary_write);
        compiler.WriteCache(ko.Stream(), binary_write);
    }

    if (num_success != 0) return 0;
    else return 1;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
