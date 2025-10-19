# Contributors

This project is a derivative work based on TensorLight, with substantial modifications and enhancements to create Tofu.

## Original Author

**Zhao Zhixu** (2018-2020)
- Created TensorLight, the original tensor computation library
- Core tensor operations (basic forward passes)
- Original build system foundation

## Tofu Development (2025)

**cakula / Vasthu**

Major contributions and enhancements:
- **Dynamic Computation Graph**: Complete graph infrastructure implementation
  - Graph nodes with forward/backward pass tracking
  - Parameter management and gradient accumulation
  - Graph traversal for automatic differentiation
  - Memory-efficient graph operations

- **Automatic Differentiation**: Implemented backward passes for all essential operations
  - Element-wise multiply (MUL) gradient
  - Transpose gradient with permutation handling
  - MSE loss gradient for regression
  - Cross-entropy loss gradient with numerical stability
  - Layer normalization gradient with mean/variance correction

- **Comprehensive Validation** (13+ tests):
  - Phase 1: Gradient correctness (6 tests with numerical validation)
  - Phase 2: Architecture diversity (5 tests: residual networks, deep networks)
  - Phase 3: Problem diversity (2 tests: multi-class classification, regression)
  - Edge case test suite (7 tests documenting behavior for zero values, NaN/Inf, extreme values)

- **CI/CD Infrastructure**:
  - GitHub Actions workflow with multi-platform testing (Ubuntu, macOS)
  - AddressSanitizer integration for memory safety verification
  - Automated builds and test reporting

- **Examples**:
  - CNN training example (8x8 pattern recognition, 100% accuracy)
  - ResNet training example (residual blocks with skip connections, 100% accuracy)

- **Build System & Tooling**:
  - ESP32 cross-compilation support (configure script with --esp32 option)
  - Enhanced build system with platform detection
  - pkg-config integration

- **Documentation**:
  - CHANGELOG.md with Keep a Changelog format
  - Comprehensive validation documentation (VALIDATION_PLAN.md, PHASE2_RESULTS.md)
  - Roadmap to v1.0.0 (ROADMAP_TO_V1.md)
  - Error handling and limitations documentation
  - Milestone implementation plans

- **Quality Assurance**:
  - Memory safety verification (zero leaks detected with AddressSanitizer)
  - Edge case documentation and regression testing
  - Performance baseline establishment

- **Project Rebranding**: Renamed to Tofu with updated branding and documentation

## Acknowledgments

We are grateful to Zhao Zhixu for creating TensorLight, which provided the foundational tensor operations that Tofu builds upon. The original tensor computation library served as a starting point for developing the complete deep learning framework.

## Contributing

We welcome contributions! Please see our GitHub repository for contribution guidelines.
