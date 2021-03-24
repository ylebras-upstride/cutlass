#include <cuda_runtime.h>
#include <iostream>
#include "cutlass/tensor_ref.h"
#include "cutlass/conv/conv2d_problem_size.h"
#include "cutlass/util/reference/device/convolution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/convolution.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/tensor_view_io.h"

using namespace cutlass;
using namespace conv;

using T = cutlass::Quaternion<float>;
using ElementA           = T;
using ElementB           = T;
using ElementC           = T;
using ElementAccumulator = T;
using ElementCompute     = T;
using Layout = cutlass::layout::TensorNHWC;

// Command line options parsing
struct Options {

  bool help;
  cutlass::Tensor4DCoord inputShape;
  cutlass::Tensor4DCoord filterShape;
  cutlass::Tensor4DCoord padding;
  cutlass::MatrixCoord conv_stride;
  cutlass::MatrixCoord dilation;
  bool reference_check;
  bool measure_performance;
  int iterations;
  bool save_workspace;
  T alpha;
  T beta;
  bool benchmark;
  std::string tag;

  Options():
    help(false),
    inputShape(1, 32, 32, 32),
    filterShape(32, 3, 3, 32),
    padding(1, 1, 1, 1),
    conv_stride(1, 1),
    dilation(1, 1),
    reference_check(false),
    measure_performance(true),
    iterations(20),
    save_workspace(false),
    alpha(1,1,1,1),
    beta(0,0,0,0),
    benchmark(false) { }

  // Verify the problem size is compatible with the CUTLASS Convolution implementation.
  bool valid() {

    //
    // CUTLASS attempts to load 128b vectors of int4b_t elements. Consequently,
    // all pointers, strides, and tensor extents must be divisible by 32 elements.
    //
    int const kAlignment = 32;

    if ((inputShape.c() % kAlignment) ||
      (filterShape.n() % kAlignment)) {

      // misaligned tensors
      return false;
    }

    // Invalid padding
    if ((padding.h() != filterShape.h() / 2) ||
      (padding.w() != filterShape.w() / 2)) {

      return false;
    }

    return true;
  }

  /// Updates input and filter sizes
  void update(
    cutlass::Tensor4DCoord inputShape,
    cutlass::Tensor4DCoord filterShape) {

    this->inputShape = inputShape;
    this->filterShape = filterShape;

    padding.n() = filterShape.h() / 2;
    padding.h() = filterShape.h() / 2;
    padding.w() = filterShape.w() / 2;
    padding.c() = filterShape.w() / 2;
  }

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
    }

    if (cmd.check_cmd_line_flag("ref-check")) {
      reference_check = true;
    }

    if (cmd.check_cmd_line_flag("perf-check")) {
      measure_performance = true;
    }

    if (cmd.check_cmd_line_flag("save-workspace")) {
      save_workspace = true;
    }

    if (cmd.check_cmd_line_flag("benchmark")) {
      benchmark = true;
    }

    cmd.get_cmd_line_argument("n", inputShape.n());
    cmd.get_cmd_line_argument("h", inputShape.h());
    cmd.get_cmd_line_argument("w", inputShape.w());
    cmd.get_cmd_line_argument("c", inputShape.c());

    cmd.get_cmd_line_argument("k", filterShape.n());
    cmd.get_cmd_line_argument("r", filterShape.h());
    cmd.get_cmd_line_argument("s", filterShape.w());
    filterShape.c() = inputShape.c();

    cmd.get_cmd_line_argument("alpha", alpha);
    cmd.get_cmd_line_argument("beta", beta);

    cmd.get_cmd_line_argument("iterations", iterations);
    cmd.get_cmd_line_argument("tag", tag);

    if (filterShape.h() == 3 && filterShape.w() == 3) {
      padding = {1, 1, 1, 1};
    }
    else {
      filterShape.h() = 1;
      filterShape.w() = 1;
      padding = {0, 0, 0, 0};
    }
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "19_conv2d_fprop_quat example\n\n"
      << "  This example uses cutlass::Quaternion<float> to compute\n"
      << "  forward convolution on tensors of layout NHWC.\n\n"
      << "Options:\n\n"
      << "  --help               If specified, displays this usage statement.\n\n"
      << "  --n <int>            Input tensor extent N\n"
      << "  --h <int>            Input tensor extent H\n"
      << "  --w <int>            Input tensor extent W\n"
      << "  --c <int>            Input tensor extent C\n"
      << "  --k <int>            Filter extent K\n"
      << "  --r <int>            Filter extent R\n"
      << "  --s <int>            Filter extent S\n\n"
      << "  --alpha <float>      Epilogue scalar alpha\n"
      << "  --beta <float>       Epilogue scalar beta\n\n"
      << "  --ref-check          If set (true), reference check on the host is computed\n"
      << "  --perf-check         If set (true), performance is measured.\n"
      << "  --benchmark          If set (true), performance benchmarking on several layers and batch-size.\n"
      << "  --iterations <int>   Number of profiling iterations to perform.\n"
      << "  --save-workspace     If set, workspace is written to a text file.\n"
      << "  --tag <string>       String to replicate across the first column in the results table\n";

    out << "\n\nExamples:\n\n"
      << "$ ./examples/19_conv2d_fprop_quat/19_conv2d_fprop_quat  --n=32 --h=224 --w=224 --c=128 --k=256 --r=1 --s=1\n\n"
      << "$ ./examples/19_conv2d_fprop_quat/19_conv2d_fprop_quat  --n=1 --h=224 --w=224 --c=32 --k=32 --r=3 --s=3 --ref-check\n\n";

    return out;
  }

  /// Computes the output tensor size (NPQK)
  cutlass::Tensor4DCoord outputShape() const {
    return cutlass::Tensor4DCoord(
      inputShape.n(),
      (inputShape.h() + padding.n() + padding.h() - filterShape.h()) / conv_stride.row() + 1,
      (inputShape.w() + padding.w() + padding.c() - filterShape.w()) / conv_stride.column() + 1,
      filterShape.n());
  }

  /// Compute performance in GFLOP/s
  /// TODO: Consider changing formulas for quaternion computation, not sure if this is working too.
  double gflops(double runtime_s) const {

    // Number of multiply-adds = NPQK * CRS
    int64_t fmas = outputShape().product() * int64_t(filterShape.h() * filterShape.w() * filterShape.c());

    // Sixteen operation per quaternion multiplication and two flops per multiply-add
    return 16.0 * 2.0 * double(fmas) / double(1.0e9) / runtime_s;
  }
};

struct Result {
  double runtime_ms;
  double gflops;
  cutlass::Status status;
  cutlass::Status reference_check;
  cudaError_t error;

  Result():
    runtime_ms(0),
    gflops(0),
    status(cutlass::Status::kSuccess),
    reference_check(cutlass::Status::kInvalid),
    error(cudaSuccess) { }

  static std::ostream & print_header(std::ostream &out, Options const &options) {

    if (!options.tag.empty()) {
      out << "Name,";
    }

    out << "Layer,N,H,W,C,K,R,S,Runtime,GFLOPs";

    return out;
  }

  std::ostream & print(std::ostream &out, int idx, Options const &options) {

    if (!options.tag.empty()) {
      out << options.tag << ",";
    }

    out
      << "Conv2dFprop_" << idx << ","
      << options.inputShape.n() << ","
      << options.inputShape.h() << ","
      << options.inputShape.w() << ","
      << options.inputShape.c() << ","
      << options.filterShape.n() << ","
      << options.filterShape.h() << ","
      << options.filterShape.w() << ","
      << runtime_ms << ","
      << gflops;

    return out;
  }
};

Result run_profile (const Options& options) {
    Result result;

    cutlass::Tensor4DCoord inputShape(options.inputShape.n(),options.inputShape.h(),options.inputShape.w(),options.inputShape.c());
    cutlass::Tensor4DCoord filterShape(options.filterShape.n(),options.filterShape.h(),options.filterShape.w(),options.filterShape.c());
    cutlass::Tensor4DCoord outputShape(
        inputShape.n(),
        (inputShape.h() + options.padding.n() + options.padding.h() - filterShape.h()) / options.conv_stride.row() + 1,
        (inputShape.w() + options.padding.w() + options.padding.c() - filterShape.w()) / options.conv_stride.column() + 1,
        filterShape.n());

    cutlass::HostTensor<ElementA, Layout> inputTensor(inputShape);
    cutlass::HostTensor<ElementB, Layout> kernelTensor(filterShape);
    cutlass::HostTensor<ElementC, Layout> outputTensor(outputShape);
    cutlass::HostTensor<ElementC, Layout> outputTensorRef(outputShape);

    //
    // Initialize tensors
    //
    // Fill tensor A on host with uniform-distribution random data
    cutlass::reference::host::TensorFillRandomUniform(
        inputTensor.host_view(),
        1,
        ElementA(7),
        ElementA(-8),
        0);

    // Fill tensor B on host with uniform-distribution random data
    cutlass::reference::host::TensorFillRandomUniform(
        kernelTensor.host_view(),
        1,
        ElementB(7),
        ElementB(-8),
        0);

    // Fill tensor C on host with zeros
    cutlass::reference::host::TensorFill(
        outputTensor.host_view());

    // Fill tensor C for reference on host with zeros
    cutlass::reference::host::TensorFill(
        outputTensorRef.host_view());

    // Copy data from host to GPU
    inputTensor.sync_device();
    kernelTensor.sync_device();
    outputTensor.sync_device();
    outputTensorRef.sync_device();

    //
    // Define arguments for CUTLASS Convolution
    //
    Conv2dProblemSize problemSize(
        inputShape.n(),  inputShape.h(),  inputShape.w(), inputShape.c(),
        outputShape.h(), outputShape.w(),
        filterShape.n(), filterShape.h(), filterShape.w(),
        Mode::kCrossCorrelation);

    int kThreadM = 2;     // shape of a thread's tile in the GEMM M dimension
    int kThreadN = 4;     // shape of a thread's tile in the GEMM N dimension
    int kCtaShapeM = 16;  // shape of a threadblock in units of threads
    int kCtaShapeN = 8;   // shape of a threadblock in units of threads

    int64_t npq = int64_t(problemSize.N) * problemSize.P * problemSize.Q;
    int64_t blocks_m = (npq + (kCtaShapeM * kThreadM) - 1) / (kCtaShapeM * kThreadM);

    dim3 block(kCtaShapeM, kCtaShapeN);
    dim3 grid(uint32_t(blocks_m), (problemSize.K + (kCtaShapeN * kThreadN) - 1) / (kCtaShapeN * kThreadN));
    cudaStream_t stream = nullptr;

    // Compute a first iteration
    cutlass::reference::device::kernel::Conv2dFprop<
    T, Layout,
    T, Layout,
    T, Layout,
    T>
    <<< grid, block, 0, stream >>>
      (
       problemSize,
       inputTensor.device_ref(),  // TensorRef<ElementA, LayoutA> tensorA,
       kernelTensor.device_ref(), // TensorRef<ElementB, LayoutB> tensorB,
       outputTensor.device_ref(), // TensorRef<ElementC, LayoutC> tensor_y_in,
       outputTensor.device_ref(), // TensorRef<ElementC, LayoutC> tensor_y_out,
       options.alpha,             // ElementCompute alpha,
       options.beta               // ElementCompute beta
      );

    // Check reference on host
    if (options.reference_check) {
      std::cout << "Verification on host...\n";
      cutlass::reference::host::Conv2dFprop<T, Layout,
                                            T, Layout,
                                            T, Layout,
                                            T>
                                          (problemSize,
                                           inputTensor.host_ref(),      // TensorRef<ElementA, LayoutA> tensorA,
                                           kernelTensor.host_ref(),     // TensorRef<ElementB, LayoutB> tensorB,
                                           outputTensorRef.host_ref(),  // TensorRef<ElementC, LayoutC> tensor_y_in,
                                           outputTensorRef.host_ref(),  // TensorRef<ElementC, LayoutC> tensor_y_out,
                                           options.alpha,               // ElementCompute alpha,
                                           options.beta                 // ElementCompute beta
                                          );

      // Check if output from CUTLASS kernel and reference kernel are equal or not
      outputTensor.sync_host();

      bool passed = cutlass::reference::host::TensorEquals(
        outputTensor.host_view(),
        outputTensorRef.host_view());

      if (!passed) {
        result.reference_check = cutlass::Status::kErrorInternal;
        std::cout << "ERROR - results miscompared.\n";
      }
      else {
        result.reference_check = cutlass::Status::kSuccess;
        std::cout << "Passed.\n";
      }
    } else {
      result.reference_check = cutlass::Status::kInvalid;
    }

    // write information in dat file
    if (options.save_workspace) {
      std::stringstream ss;

      ss << "20_conv_workspace_conv2d_fprop_quat_"
         << options.inputShape.n() << "x" << options.inputShape.h() << "x" << options.inputShape.w() << "x" << options.inputShape.c()
         << "_"
         << options.filterShape.n() << "x" << options.filterShape.h() << "x" << options.filterShape.w() << "x" << options.filterShape.c()
         << ".dat";

      std::ofstream output_workspace(ss.str());

      output_workspace
        << "Input = \n" << inputTensor.host_view() << "\n\n"
        << "Filters = \n" << kernelTensor.host_view() << "\n\n";

      if (options.reference_check) {
        output_workspace << "Reference = \n" << outputTensorRef.host_view() << "\n\n";
      }

      output_workspace << "Computed = \n" << outputTensor.host_view() << std::endl;

      std::cout << "Results written to '" << ss.str() << "'." << std::endl;
    }

    if (options.measure_performance) {
      cudaEvent_t events[2];

      for (auto & event : events) {
        result.error = cudaEventCreate(&event);
        if (result.error != cudaSuccess) {
          std::cerr << "cudaEventCreate() failed: " << cudaGetErrorString(result.error) << std::endl;
          return result;
        }
      }

      // Record an event at the start of a series of convolution operations.
      result.error = cudaEventRecord(events[0]);
      if (result.error != cudaSuccess) {
        std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(result.error) << std::endl;
        return result;
      }

      for (int iteration = 1; iteration < options.iterations; ++iteration) {
          //Compute
          cutlass::reference::device::kernel::Conv2dFprop<
          T, Layout,
          T, Layout,
          T, Layout,
          T>
          <<< grid, block, 0, stream >>>
            (
             problemSize,
             inputTensor.device_ref(),  // TensorRef<ElementA, LayoutA> tensorA,
             kernelTensor.device_ref(), // TensorRef<ElementB, LayoutB> tensorB,
             outputTensor.device_ref(), // TensorRef<ElementC, LayoutC> tensor_y_in,
             outputTensor.device_ref(), // TensorRef<ElementC, LayoutC> tensor_y_out,
             options.alpha,             // ElementCompute alpha,
             options.beta               // ElementCompute beta
            );
      }

      // Record an event when the convolutions have been launched.
      result.error = cudaEventRecord(events[1]);
      if (result.error != cudaSuccess) {
        std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(result.error) << std::endl;
        return result;
      }

      // Wait for work on the device to complete.
      result.error = cudaEventSynchronize(events[1]);
      if (result.error != cudaSuccess) {
        std::cerr << "cudaEventSynchronize() failed: " << cudaGetErrorString(result.error) << std::endl;
        return result;
      }

      // Measure elapsed runtime
      float runtime_ms = 0;
      result.error = cudaEventElapsedTime(&runtime_ms, events[0], events[1]);
      if (result.error != cudaSuccess) {
        std::cerr << "cudaEventElapsed() failed: " << cudaGetErrorString(result.error) << std::endl;
        return result;
      }
      result.runtime_ms = double(runtime_ms) / double(options.iterations);

      // Cleanup
      for (auto event : events) {
          (void)cudaEventDestroy(event);
      }
    }
    return result;
}


int main(int argc, char const **args) {
    Options options;
    options.parse(argc, args);

    if (options.help) {
      options.print_usage(std::cout) << std::endl;
      return 0;
    }

    if (options.benchmark) {
      int batch_sizes[] = {1, 32, 64, 128}; //, 256, 512};
      struct Benchmark {
          int h, w, c, k, r, s;
        } layers[] = {
          {56,  56,   64,   256, 1, 1},
          {56,  56,   64,    64, 1, 1},
          {56,  56,   64,    64, 3, 3},
          {56,  56,  256,    64, 1, 1},
          {56,  56,  256,   512, 1, 1},
          {56,  56,  256,   128, 1, 1},
          {28,  28,  128,   128, 3, 3},
          {28,  28,  128,   512, 1, 1},
          {28,  28,  512,   128, 1, 1},
          {28,  28,  512,  1024, 1, 1},
          {28,  28,  512,   256, 1, 1},
          {14,  14,  256,   256, 3, 3},
          {14,  14,  256,  1024, 1, 1},
          {14,  14,  1024,  256, 1, 1},
          {14,  14,  1024, 2048, 1, 1},
          {14,  14,  1024,  512, 1, 1},
          {7,    7,   512,  512, 3, 3},
        };
      int idx = 0;
      
      Result result;
      Result::print_header(std::cout, options) << std::endl;
      for (auto const &layer : layers) {
          for (auto N : batch_sizes) {
            options.update({N, layer.h, layer.w, layer.c}, {layer.k, layer.r, layer.s, layer.c});
            result = run_profile(options);

            result.print(std::cout, 1, options) << std::endl;
          }
          ++idx;
      }
    } else {
      // One unique run
      Result result = run_profile(options);
      Result::print_header(std::cout, options) << std::endl;
      result.print(std::cout, 1, options) << std::endl;
    }
}
