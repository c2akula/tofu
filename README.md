# TensorLight
TensorLight is a light-weight tensor operation library for C.

## Prerequisites
The following steps have been tested for Ubuntu 16.04 but should work with
other distros as well. 
Required packages can be installed using the following command:

```
sudo apt-get install build-essential perl git pkg-config check
```

### ESP32 Cross-Compilation
To cross-compile for ESP32, you need the ESP32 toolchain installed. You can install it by:

1. Installing ESP-IDF: Follow the instructions at [ESP-IDF Programming Guide](https://docs.espressif.com/projects/esp-idf/en/latest/esp32/get-started/index.html)
2. Ensuring the ESP32 toolchain is in your PATH or specify its directory with the `--esp32-toolchain-dir` option

### Docker Testing Environment
For consistent testing across different environments, you can use the provided Docker setup:

1. Make sure you have Docker installed on your system
2. Run the provided script to build and test in a Docker container:
   ```
   ./run-tests-docker.sh
   ```

This will create an Ubuntu 22.04 container with all required dependencies and run the tests inside it. See the [Docker documentation](docker/README.md) for more information.

## Building and Installation
1.  Clone this repository to your local directory.

    ```
    cd <my_working_directory>
    git clone https://github.com/zhaozhixu/TensorLight.git
    cd TensorLight
    ```

2.  Build and install

    First, configure your installation using:
    
    ```
    chmod +x configure
    ./configure
    ```
    There are options to customize your building and installation process.
    You can append them after `./configure`. 
    
    For cross-compiling for ESP32:
    ```
    ./configure --esp32=yes
    ```
    
    Or if your ESP32 toolchain is not in your PATH:
    ```
    ./configure --esp32=yes --esp32-toolchain-dir=/path/to/esp-idf/tools/xtensa-esp32-elf
    ```
    
    Detailed `./configure` options can be printed using `./configure -h`.

    After that, use `make` to compile the library and run the tests. Then `make install`
    to copy the library files and headers into the installation directory,
    or `sudo make install` if you don't have the permissions with that directory.
    
    ```
    make
    sudo make install
    ```

3.  Other `make` options

    Use `make info` to see other `make` options.
    Especially, you can use `make clean` to clean up the build directory and all
    object files, and `make uninstall` to remove library files and headers from
    the installation directory.

## Usage
Include `tl_tensor.h` in your project to use TensorLight functions.

Documentation is coming soon. But it should be familiar if you have experience
with `numpy` in Python.

You can use the following command to get the compilation and linking flags when
building your project.

```
pkg-config --cflags --libs tensorlight
```
