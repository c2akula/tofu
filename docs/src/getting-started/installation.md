# Installation

Learn how to build and install Tofu on Linux, macOS, and ESP32 microcontrollers. This guide covers everything from downloading the source code to verifying your installation works correctly.

**In this guide you'll learn:**
- Prerequisites and system requirements for each platform
- How to configure and build Tofu from source
- How to run tests to verify your build
- How to install Tofu to your system
- How to cross-compile for ESP32 embedded systems

**Prerequisites:**
- A C compiler (GCC 7+ or Clang 10+)
- GNU Make build system
- Git version control
- pkg-config package manager

**Estimated time:** 10 minutes for basic installation, 15 minutes including tests

---

## Quick Installation

If you're already familiar with building C projects, here's the fastest path to get Tofu running:

```bash
# Clone the repository
cd ~/projects  # or your preferred workspace
git clone https://github.com/c2akula/tofu.git
cd tofu

# Configure and build
chmod +x configure
./configure
make

# Optional: Run tests to verify the build
make test

# Install to your system
sudo make install

# Verify installation
pkg-config --modversion tofu
```

After installation, you can include Tofu in your projects with:

```bash
gcc -o myprogram myprogram.c $(pkg-config --cflags --libs tofu)
```

---

## Detailed Installation - Linux

This section covers step-by-step installation on Ubuntu 20.04 and similar Linux distributions.

### System Requirements

Tofu requires the following packages on Linux:

- **Build tools**: GCC (7+), GNU Make, GNU Autotools utilities
- **Development headers**: Standard C library headers
- **Package management**: pkg-config for dependency tracking
- **Testing** (optional): Check unit testing framework

### Installing Dependencies

On Ubuntu and Debian-based distributions:

```bash
sudo apt-get update
sudo apt-get install build-essential perl git pkg-config check
```

This installs:
- `gcc` and `g++`: C/C++ compilers
- `make`: Build automation
- `perl`: Required by the configure script
- `git`: Version control
- `pkg-config`: Package configuration utility
- `check`: Unit testing framework (optional but recommended)

### Cloning the Repository

```bash
cd ~/projects  # Choose your workspace location
git clone https://github.com/c2akula/tofu.git
cd tofu
```

The repository is approximately 5 MB and includes:
- Source code in `src/` directory
- Tests in `test/` directory
- Examples in `examples/` directory
- Documentation in `docs/` directory

### Configuring the Build

Before building, configure the build system:

```bash
chmod +x configure
./configure
```

You'll see output like:

```
configure tofu version 1.0.0
Checking autotools...
Checking pkg-config...
Checking C compiler...
Checking math library...
Generated config.mk
```

For custom installation directories, use:

```bash
./configure --install-dir=/opt/tofu
./configure --prefix=$HOME/.local
```

To see all available options:

```bash
./configure -h
```

Available options include:
- `--build-dir=DIR`: Where to place build artifacts (default: `build`)
- `--install-dir=DIR`: Where to install Tofu (default: `/usr/local`)
- `--debug=yes`: Include debug symbols for development
- `--esp32=yes`: Configure for ESP32 cross-compilation

### Building the Library

After configuration, compile Tofu:

```bash
make lib
```

This builds both static and shared libraries. Expected output:

```
Compiling src/tofu_tensor.c
Compiling src/tofu_graph.c
Compiling src/tofu_optimizer.c
...
Linking build/lib/libtofu.so.1.0.0
Created: build/lib/libtofu.a
Created: build/lib/libtofu.so -> libtofu.so.1.0.0
```

Build artifacts are placed in the `build/` directory:
- `build/lib/libtofu.a`: Static library
- `build/lib/libtofu.so.1.0.0`: Shared library (versioned)
- `build/include/tofu/`: Public header files

### Running Tests

Verify your build with the comprehensive test suite:

```bash
make test
```

This builds and runs all tests. Expected output:

```
Compiling test/test_tensor.c
Compiling test/test_graph.c
Compiling test/test_optimizer.c
...
Running tests...
[====] Completed: 13 test(s), passed: 13, failed: 0
```

**All tests should pass (66 checks across 5 test suites).** If any fail:
1. Check that all dependencies are installed
2. Run `make clean` and retry
3. Report the issue with your system details

To run individual tests:

```bash
cd test
../build/test/test_tofu tensor_creation
```

### Installing to Your System

Once tests pass, install Tofu to your system:

```bash
sudo make install
```

This installs:
- Headers to `/usr/local/include/tofu/`
- Static library to `/usr/local/lib/libtofu.a`
- Shared library to `/usr/local/lib/libtofu.so.1.0.0`
- Package config file to `/usr/local/lib/pkgconfig/tofu.pc`

If you configured with a custom prefix:

```bash
# Install to home directory (no sudo needed)
./configure --prefix=$HOME/.local
make
make install
```

### Verifying Installation

Check that Tofu is installed correctly:

```bash
pkg-config --modversion tofu
```

Expected output:
```
1.0.0
```

Get compilation and linking flags:

```bash
pkg-config --cflags --libs tofu
```

Expected output:
```
-I/usr/local/include/tofu -L/usr/local/lib -ltofu
```

---

## Detailed Installation - macOS

This section covers installation on macOS 13 and newer.

### System Requirements

Tofu requires:
- **Xcode Command Line Tools**: Apple's development toolchain with GCC/Clang
- **Homebrew** (optional): Package manager for additional tools
- **pkg-config**: For dependency management

### Installing Xcode Command Line Tools

First, install the Xcode Command Line Tools:

```bash
xcode-select --install
```

A dialog will appear. Click **Install** and wait for completion (5-10 minutes).

Verify installation:

```bash
gcc --version
make --version
```

Expected output shows GCC (Apple Clang) version 14+.

### Installing Additional Dependencies

Using Homebrew to install the check testing framework:

```bash
brew install pkg-config check
```

If Homebrew isn't installed, see https://brew.sh

### Cloning and Building

The build process is identical to Linux:

```bash
cd ~/Projects  # macOS convention
git clone https://github.com/c2akula/tofu.git
cd tofu

chmod +x configure
./configure
make lib
make test
```

### Installing to Your System

Install to `/usr/local` (default) or your home directory:

```bash
# System-wide installation
sudo make install

# Or to home directory (no sudo needed)
./configure --prefix=$HOME/.local
make
make install
```

### macOS-Specific Notes

- **M1/M2 Arm chips**: Tofu builds natively on Apple Silicon. No special configuration needed.
- **Intel Macs**: Fully supported. Tofu uses standard C that compiles on both architectures.
- **Shared library permissions**: You may need to adjust Gatekeeper policies if running compiled binaries. See Apple's code signing documentation.
- **Homebrew path**: If using Homebrew, you may need `export PATH="/opt/homebrew/bin:$PATH"` on M1/M2 Macs.

---

## Cross-Compilation - ESP32

Build Tofu for embedded deployment on ESP32 microcontrollers.

### Why ESP32?

ESP32 is a popular microcontroller combining:
- WiFi and Bluetooth connectivity
- 240 MHz dual-core processor
- 520 KB RAM
- Capable of running lightweight neural networks for inference

Tofu's lightweight C implementation makes it ideal for ESP32 deployment.

### Prerequisites

You need the ESP-IDF (Espressif IoT Development Framework) toolchain:

1. **Install ESP-IDF** following the official guide:
   https://docs.espressif.com/projects/esp-idf/en/latest/esp32/get-started/index.html

2. **Verify toolchain is available:**
   ```bash
   which xtensa-esp32-elf-gcc
   ```

   Should show the path to the toolchain. If not, add to your shell profile:
   ```bash
   export PATH="$PATH:/path/to/esp-idf/tools/xtensa-esp32-elf/bin"
   ```

3. **Verify the toolchain works:**
   ```bash
   xtensa-esp32-elf-gcc --version
   ```

### Configuring for ESP32

Clone Tofu and configure for cross-compilation:

```bash
git clone https://github.com/c2akula/tofu.git
cd tofu

chmod +x configure
./configure --esp32=yes
```

If your ESP32 toolchain is not in your PATH:

```bash
./configure --esp32=yes --esp32-toolchain-dir=/opt/esp-idf/tools/xtensa-esp32-elf
```

The configure output will show:

```
configure tofu version 1.0.0
Configuring for ESP32 cross-compilation
Using toolchain: /path/to/xtensa-esp32-elf
Generated config.mk
```

### Building for ESP32

Build the library for ESP32:

```bash
make lib
```

This creates:
- `build/lib/libtofu.a`: Static library for ESP32

**Note:** Tests are skipped during ESP32 cross-compilation:
```
Skipping tests when cross-compiling for ESP32
```

This is expected. Tests require POSIX system calls not available on ESP32.

### Using Tofu on ESP32

Once built, link the static library into your ESP32 project:

1. Copy the built library:
   ```bash
   cp build/lib/libtofu.a /path/to/your/esp32-project/components/tofu/lib/
   cp -r build/include/tofu/* /path/to/your/esp32-project/components/tofu/include/
   ```

2. In your ESP32 project's CMakeLists.txt:
   ```cmake
   set(COMPONENT_LIBS tofu m c)  # Link tofu library
   ```

3. Include headers in your code:
   ```c
   #include "tofu_tensor.h"
   #include "tofu_graph.h"
   ```

4. Build and deploy:
   ```bash
   idf.py build
   idf.py flash
   ```

---

## Verification

After installation, verify everything works by running a simple test program.

### Using pkg-config

Verify Tofu is discoverable by pkg-config:

```bash
pkg-config --list-all | grep tofu
```

Should show:
```
tofu - A light-weight neural network compiler for different software/hardware backends.
```

### Using pkg-config in Your Build

When building applications that use Tofu:

```bash
gcc -o myapp myapp.c $(pkg-config --cflags --libs tofu)
```

This automatically includes:
- Correct compiler flags: `-I/usr/local/include/tofu`
- Linker flags: `-L/usr/local/lib -ltofu`

### Example Verification Program

Create a simple C program to test your installation:

```c
// test_tofu.c
#include "tofu_tensor.h"
#include <stdio.h>

int main() {
    // Create a simple tensor [2, 3]
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    int dims[] = {2, 3};
    tofu_tensor *t = tofu_tensor_create(data, 2, dims, TOFU_FLOAT);

    if (t == NULL) {
        printf("Failed to create tensor\n");
        return 1;
    }

    printf("Tensor created successfully\n");
    printf("Shape: [%d, %d]\n", t->dims[0], t->dims[1]);
    printf("Size: %d elements\n", t->len);

    tofu_tensor_free(t);
    printf("Tensor freed successfully\n");
    return 0;
}
```

Compile and run:

```bash
gcc -o test_tofu test_tofu.c $(pkg-config --cflags --libs tofu)
./test_tofu
```

Expected output:
```
Tensor created successfully
Shape: [2, 3]
Size: 6 elements
Tensor freed successfully
```

### Troubleshooting Common Issues

**Issue: `pkg-config: command not found`**
- Solution: Install pkg-config (`sudo apt-get install pkg-config` on Linux, `brew install pkg-config` on macOS)

**Issue: `fatal error: tofu_tensor.h: No such file or directory`**
- Solution: Run `make install` to install headers. Or use the pkg-config output in your compilation flags.

**Issue: `undefined reference to 'tofu_tensor_create'`**
- Solution: Ensure you're using `$(pkg-config --libs tofu)` in your linker flags

**Issue: Tests fail with "check not found"**
- Solution: Install check (`sudo apt-get install check` on Linux, `brew install check` on macOS)

**Issue: Configure fails with unknown compiler**
- Solution: Ensure GCC/Clang is installed and in your PATH. Run `gcc --version` to verify.

**Issue: Build fails on macOS with permission errors**
- Solution: Try `sudo make install` or configure with a home directory prefix: `./configure --prefix=$HOME/.local`

### Uninstallation

To remove Tofu from your system:

```bash
sudo make uninstall
```

This removes all installed files, headers, and pkg-config configuration.

If you installed to a home directory:

```bash
make uninstall
```

---

## Next Steps

Now that Tofu is installed, explore:
- **[Quick Start](../quick-start.md)**: Write your first neural network
- **[First Network](../first-network.md)**: Build a complete training example
- **[API Reference](/docs/api-reference/tensor-api.md)**: Full API documentation

Happy building!
