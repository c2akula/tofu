# Check if we're cross-compiling for ESP32
ifeq ($(ESP32), yes)
  ifeq ($(ESP32_TOOLCHAIN_DIR),)
    # No toolchain dir specified, use tools from PATH
    CC = xtensa-esp32-elf-gcc
    CXX = xtensa-esp32-elf-g++
    AR = xtensa-esp32-elf-ar cr
  else
    # Use toolchain from specified directory
    CC = $(ESP32_TOOLCHAIN_DIR)/bin/xtensa-esp32-elf-gcc
    CXX = $(ESP32_TOOLCHAIN_DIR)/bin/xtensa-esp32-elf-g++
    AR = $(ESP32_TOOLCHAIN_DIR)/bin/xtensa-esp32-elf-ar cr
  endif
else
  # Regular build
  CC ?= gcc
  CXX ?= g++
  AR = ar cr
endif

ECHO = @echo
SHELL = /bin/sh

ifdef VERBOSE
AT =
else
AT = @
endif

# Set base compiler flags
CFLAGS += -Wall -std=gnu99

# ESP32-specific flags
ifeq ($(ESP32), yes)
  # ESP32 specific flags
  CFLAGS += -DESP32 -mlongcalls -Wno-frame-address
  CXXFLAGS += -std=c++11 -Wall -DESP32 -mlongcalls -Wno-frame-address
  
  # Don't use GNU extensions on ESP32
  # Don't use pkg-config for ESP32
else
  # Regular build flags
  CFLAGS += -D_GNU_SOURCE
  CXXFLAGS += -std=c++11 -Wall
  
  INCPATHS += -I/usr/local/include
  LDFLAGS += -L/usr/local/lib -lm
  # cannot use ifeq/ifneq because they expand immediately
  INCPATHS += $(if $(REQUIRES),`pkg-config --cflags '$(REQUIRES)'`)
  LDFLAGS += $(if $(REQUIRES),`pkg-config --libs '$(REQUIRES)'`)
endif

INCPATHS += -I.
LDFLAGS += $(CFLAGS)

ifeq ($(DEBUG), yes)
CFLAGS += -g -O0 -D$(ABBR)_DEBUG
CXXFLAGS += -g -O0 -D$(ABBR)_DEBUG
LDFLAGS += -g -O0
else
CFLAGS += -O2
CXXFLAGS += -O2
LDFLAGS += -O2
endif

SRC = $(filter-out %cuda.c %cuda.cc %cuda.cpp %cudnn.c %cudnn.cc %cudnn.cpp %tensorrt.c %tensorrt.cc %tensorrt.cpp %dpu.c %dpu.cc %dpu.cpp %.cu, $(SRC))

OBJDIR = $(BUILD_DIR)/$(notdir $(CURDIR))
OBJS   = $(patsubst %.c,$(OBJDIR)/%.o,$(filter %.c,$(SRC)))
OBJS  += $(patsubst %.cc,$(OBJDIR)/%.o,$(filter %.cc,$(SRC)))
OBJS  += $(patsubst %.cpp,$(OBJDIR)/%.o,$(filter %.cpp,$(SRC)))

CFLAGS += $(INCPATHS)
CXXFLAGS += $(INCPATHS)

CFLAGS += -fPIC -fvisibility=hidden
CXXFLAGS += -fPIC -fvisibility=hidden
LDFLAGS_SO += $(LDFLAGS) -shared
ifeq ($(UNAME_S),Linux)
LDFLAGS_SO += -Wl,--no-undefined
else ifeq ($(UNAME_S),Darwin)
LDFLAGS_SO += -Wl,-undefined,error
endif
CFLAGS += $(SRC_EXTRA_FLAGS)

define concat
$1$2$3$4$5$6$7$8
endef

define make-depend-c
$(AT)$(CC) -MM -MF $(subst .o,.d,$@) -MP -MT $@ $(CFLAGS) $<
endef

define make-depend-cxx
$(AT)$(CXX) -MM -MF $(subst .o,.d,$@) -MP -MT $@ $(CXXFLAGS) $<
endef

GEN_CMD_FILE := no
ifeq ($(MAKECMDGOALS), cmd)
GEN_CMD_FILE := yes
endif
CMD_FILE ?= $(BUILD_DIR)/compile_commands.json

ifeq ($(GEN_CMD_FILE), no)
define compile-c
$(ECHO) "  CC\t" $@
$(call make-depend-c)
$(AT)$(CC) $(CFLAGS) -c -o $@ $<
endef
else
define compile-c
$(ECHO) "  GEN\t" $(CMD_FILE) for $@
$(AT)$(BUILDTOOLS_DIR)/gen_compile_commands.pl -f $(CMD_FILE) `pwd` $< "$(CC) $(CFLAGS) -c -o $@ $<"
endef
endif

ifeq ($(GEN_CMD_FILE), no)
define compile-cxx
$(ECHO) "  CXX\t" $@
$(call make-depend-cxx)
$(AT)$(CXX) $(CXXFLAGS) -c -o $@ $<
endef
else
define compile-cxx
$(ECHO) "  GEN\t" $(CMD_FILE) for $@
$(AT)$(BUILDTOOLS_DIR)/gen_compile_commands.pl -f $(CMD_FILE) `pwd` $< "$(CXX) $(CXXFLAGS) -c -o $@ $<"
endef
endif

# CUDA support removed

define ld-bin
$(ECHO) "  LD\t" $@
$(AT)$(CC) -o $@ $^ $(LDFLAGS)
endef

define ld-so
$(ECHO) "  LD\t" $@
$(AT)$(CC) -o $@ $^ $(LDFLAGS_SO)
endef

define ar-a
$(ECHO) "  AR\t" $@
$(AT)$(AR) $@ $^
endef
