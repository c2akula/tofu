# Makefile for Bug Reproduction Tests
# These tests demonstrate the critical bugs found in the code review

CC = gcc
CFLAGS = -Wall -Wextra -std=c99 -g
LDFLAGS = -lm

# Test executables
TESTS = test_stride_bug \
        test_fprintf_bug \
        test_comparison_bug \
        test_integer_overflow_bug \
        test_ndebug_logic

# Default target - build all tests
all: $(TESTS)
	@echo ""
	@echo "═══════════════════════════════════════════════════════"
	@echo "  All bug reproduction tests built successfully!"
	@echo "═══════════════════════════════════════════════════════"
	@echo ""
	@echo "Run 'make run' to execute all tests"
	@echo "Run individual tests with: ./test_<name>"
	@echo ""

# Build individual tests
test_stride_bug: test_stride_bug.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

test_fprintf_bug: test_fprintf_bug.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

test_comparison_bug: test_comparison_bug.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

test_integer_overflow_bug: test_integer_overflow_bug.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

test_ndebug_logic: test_ndebug_logic.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# Run all tests
run: all
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════╗"
	@echo "║          RUNNING ALL BUG REPRODUCTION TESTS                  ║"
	@echo "╚══════════════════════════════════════════════════════════════╝"
	@echo ""
	@for test in $(TESTS); do \
		echo ""; \
		echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; \
		echo "Running: $$test"; \
		echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; \
		./$$test; \
		echo ""; \
	done
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════╗"
	@echo "║          ALL BUG REPRODUCTION TESTS COMPLETED                ║"
	@echo "╚══════════════════════════════════════════════════════════════╝"
	@echo ""

# Run individual tests
run-stride: test_stride_bug
	./test_stride_bug

run-fprintf: test_fprintf_bug
	./test_fprintf_bug

run-comparison: test_comparison_bug
	./test_comparison_bug

run-overflow: test_integer_overflow_bug
	./test_integer_overflow_bug

run-ndebug: test_ndebug_logic
	./test_ndebug_logic

# Also build with NDEBUG defined to test release mode behavior
test_ndebug_logic_release: test_ndebug_logic.c
	$(CC) $(CFLAGS) -DNDEBUG -o test_ndebug_logic_release $< $(LDFLAGS)

run-ndebug-release: test_ndebug_logic_release
	@echo ""
	@echo "Running NDEBUG test in RELEASE mode (with -DNDEBUG):"
	@echo ""
	./test_ndebug_logic_release

# Clean up
clean:
	rm -f $(TESTS) test_ndebug_logic_release
	@echo "Cleaned up all test executables"

# Help target
help:
	@echo "Bug Reproduction Test Suite"
	@echo "============================="
	@echo ""
	@echo "Targets:"
	@echo "  make           - Build all tests"
	@echo "  make run       - Build and run all tests"
	@echo "  make clean     - Remove all test executables"
	@echo ""
	@echo "Individual test targets:"
	@echo "  make run-stride      - Test stride calculation bug"
	@echo "  make run-fprintf     - Test fprintf type cast bugs"
	@echo "  make run-comparison  - Test comparison function bugs"
	@echo "  make run-overflow    - Test integer overflow bugs"
	@echo "  make run-ndebug      - Test NDEBUG logic bug"
	@echo ""
	@echo "Special targets:"
	@echo "  make run-ndebug-release - Test NDEBUG bug in release mode"
	@echo ""

.PHONY: all run clean help run-stride run-fprintf run-comparison run-overflow run-ndebug run-ndebug-release
