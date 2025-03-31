# TensorLight Development Guide

## Build Commands
- `./configure` - Configure build (run `./configure -h` for options)
- `./configure --esp32=yes` - Configure for ESP32 cross-compilation
- `./configure --esp32=yes --esp32-toolchain-dir=PATH` - Configure with custom ESP32 toolchain path
- `make` or `make lib` - Build libraries
- `make test` - Build and run all tests (skipped when cross-compiling for ESP32)
- `make -C test bin` - Build tests without running
- `cd test && ../build/test/test_tensorlight [test_name]` - Run specific test
- `make cmd` - Generate compile_commands.json for tooling
- `make clean` - Remove build files
- `make install` - Install to configured directory

## Code Style Guidelines
- **Formatting**: 4-space indentation, K&R brace style
- **Naming**: snake_case for functions/variables, prefix with `tl_`
- **Error Handling**: Use `assert()` for internal validation
- **Types**: Use explicit types from `tl_type.h` (TL_INT32, TL_FLOAT, etc.)
- **Headers**: Include guards with `_H` suffix
- **Documentation**: Use C-style comments with function descriptions
- **Memory**: Always free allocated resources, check for NULL
- **Exports**: Use TL_EXPORT macro for public API functions
- **Testing**: Every function should have corresponding test in test/ directory