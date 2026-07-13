CXX := g++
CC := gcc
TARGET := MagXTK-7SM2

BUILD_PROFILE ?= auto
PGO ?= 0
BUILD_DIR := build
TARGET_TRIPLE := $(shell $(CXX) -dumpmachine)
X86_64_TARGET := $(filter x86_64% amd64%,$(TARGET_TRIPLE))

ifeq ($(X86_64_TARGET),)
  $(error MagXTK-7SM2 64-bit profiles require an x86_64 compiler target; got '$(TARGET_TRIPLE)')
endif

ifeq ($(BUILD_PROFILE),auto)
  EFFECTIVE_PROFILE := native64
else ifeq ($(BUILD_PROFILE),portable64)
  EFFECTIVE_PROFILE := portable64
else ifeq ($(BUILD_PROFILE),native64)
  EFFECTIVE_PROFILE := native64
else ifeq ($(BUILD_PROFILE),avx2+bmi2)
  EFFECTIVE_PROFILE := avx2+bmi2
else ifeq ($(BUILD_PROFILE),avx512+bmi2)
  EFFECTIVE_PROFILE := avx512+bmi2
else
  $(error Unknown BUILD_PROFILE '$(BUILD_PROFILE)'; use auto, portable64, native64, avx2+bmi2, or avx512+bmi2)
endif

ifneq ($(PGO),0)
  ifneq ($(PGO),1)
    $(error Unknown PGO value '$(PGO)'; use PGO=0 or PGO=1)
  endif
endif

# Embedded P2-family network - single source of truth for the filename.
# Changing this updates MnueEmbedded.S, Mnue.cpp, and Uci.cpp automatically.
MNUE_EMBEDDED_FILE := mm-4ac81907e.MNUE

# Fresh-clone helper: keep the large MNUE out of git.
# Linux server uses the local file first; other workers fall back to GitHub raw.
MNUE_EMBEDDED_LOCAL ?= /home/magnus/$(MNUE_EMBEDDED_FILE)
MNUE_EMBEDDED_URL ?= https://raw.githubusercontent.com/TMagnusN/MagnusMNUE/main/$(MNUE_EMBEDDED_FILE)

OBJ_VARIANT ?= normal
OBJ_DIR := $(BUILD_DIR)/obj/$(EFFECTIVE_PROFILE)/$(OBJ_VARIANT)
PGO_DIR := $(BUILD_DIR)/pgo/$(EFFECTIVE_PROFILE)
PGO_BENCH_LOG := $(PGO_DIR)/bench.log
PGO_BENCH_ARGS ?= bench

SRC := \
	board/Attack.cpp \
	search/History.cpp \
	search/Lmr.cpp \
	Main.cpp \
	search/MovePicker.cpp \
	board/MoveGen.cpp \
	search/Nmp.cpp \
	board/Position.cpp \
	search/Search.cpp \
	search/See.cpp \
	syzygy/Syzygy.cpp \
	search/Time.cpp \
	search/TT.cpp \
	Uci.cpp \
	mnue/Mnue.cpp \
	mnue/MnueX2K16PawnNetwork.cpp

FATHOM_DIR := ../Vendor/fathom
FATHOM_OBJ := $(OBJ_DIR)/FathomTbProbe.o

# Embedded network object - compiled separately without LTO / PGO to keep
# the 20.1 MiB incbin isolated from instrumentation and whole-program analysis.
MNUE_EMBEDDED_OBJ := $(OBJ_DIR)/mnue/MnueEmbedded.o

OBJ := $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(SRC)) $(FATHOM_OBJ) $(MNUE_EMBEDDED_OBJ)
DEP := $(OBJ:.o=.d)
MODULE_OBJ_DIRS := \
	$(OBJ_DIR)/board \
	$(OBJ_DIR)/mnue \
	$(OBJ_DIR)/search \
	$(OBJ_DIR)/syzygy

PORTABLE64_FLAGS := \
	-m64 \
	-march=x86-64 \
	-mtune=generic

NATIVE64_FLAGS := \
	-m64 \
	-march=native \
	-mtune=native

AVX2_BMI2_FLAGS := \
	-m64 \
	-march=x86-64-v3 \
	-mtune=generic \
	-mbmi2

AVX512_BMI2_FLAGS := \
	-m64 \
	-march=x86-64-v4 \
	-mtune=generic \
	-mbmi2

ifeq ($(EFFECTIVE_PROFILE),portable64)
	ARCH_FLAGS := $(PORTABLE64_FLAGS)
	PROFILE_CPPFLAGS := -DMAGNUS_DISABLE_PEXT=1
	FATHOM_PROFILE_FLAGS := -DTB_NO_HW_POP_COUNT=1
else ifeq ($(EFFECTIVE_PROFILE),avx512+bmi2)
	ARCH_FLAGS := $(AVX512_BMI2_FLAGS)
	PROFILE_CPPFLAGS :=
	FATHOM_PROFILE_FLAGS :=
else ifeq ($(EFFECTIVE_PROFILE),avx2+bmi2)
	ARCH_FLAGS := $(AVX2_BMI2_FLAGS)
	PROFILE_CPPFLAGS :=
	FATHOM_PROFILE_FLAGS :=
else
	ARCH_FLAGS := $(NATIVE64_FLAGS)
	PROFILE_CPPFLAGS :=
	FATHOM_PROFILE_FLAGS :=
endif

OPT_FLAGS := \
	-Ofast \
	-DNDEBUG \
	-flto=auto \
	-fuse-linker-plugin \
	-fomit-frame-pointer \
	-funroll-loops \
	-fstrict-aliasing \
	-ffast-math \
	-fno-math-errno \
	-fno-signed-zeros \
	-fno-trapping-math \
	-fdevirtualize \
	-fdevirtualize-at-ltrans \
	-ftree-vectorize \
	-fipa-pta \
	-fipa-cp \
	-fipa-cp-clone \
	-fipa-icf \
	-fipa-ra \
	-falign-functions=32 \
	-falign-loops=32 \
	-falign-jumps=32 \
	-falign-labels=32 \
	-fweb \
	-fprefetch-loop-arrays \
	-frename-registers

LOWLEVEL_FLAGS := \
	-fno-exceptions \
	-fno-rtti

ENCODING_FLAGS := \
	-finput-charset=UTF-8 \
	-fexec-charset=UTF-8

WARN_FLAGS := \
	-Wall \
	-Wextra \
	-Wshadow \
	-Wconversion \
	-Wcast-qual \
	-Wno-unused-parameter \
	-Wno-sign-conversion

CPPFLAGS := -MMD -MP -iquote . -I$(FATHOM_DIR) $(PROFILE_CPPFLAGS) \
	-DMNUE_EMBEDDED_FILENAME='"$(MNUE_EMBEDDED_FILE)"'

BASE_CXXFLAGS := \
	-std=c++20 \
	$(ARCH_FLAGS) \
	$(OPT_FLAGS) \
	$(LOWLEVEL_FLAGS) \
	$(ENCODING_FLAGS) \
	$(WARN_FLAGS) \
	$(CPPFLAGS)

FATHOM_CFLAGS := \
	-std=gnu99 \
	$(ARCH_FLAGS) \
	-O3 \
	-DNDEBUG \
	$(FATHOM_PROFILE_FLAGS) \
	-I$(FATHOM_DIR)

LINK_OPT_LEVEL ?= 3

# Stack reserve size.
# 1 GiB = 1024^3 = 1073741824 bytes.
# Do not set this to tens of TiB: each created thread may reserve stack address space,
# which can make the executable fail at runtime or make std::thread throw std::system_error.
STACK_SIZE_BYTES ?= 1073741824

ifeq ($(OS),Windows_NT)
	# MinGW / GNU ld PE stack reserve.
	STACK_LDFLAGS := -Wl,--stack,$(STACK_SIZE_BYTES)
else
	# ELF GNU ld stack-size note.
	STACK_LDFLAGS := -Wl,-z,stack-size=$(STACK_SIZE_BYTES)
endif

BASE_LDFLAGS := \
	$(ARCH_FLAGS) \
	-Ofast \
	-flto=auto \
	-Wl,-O$(LINK_OPT_LEVEL) \
	$(STACK_LDFLAGS)

PGO_GENERATE_CXXFLAGS := -fprofile-generate=$(PGO_DIR)
PGO_GENERATE_LDFLAGS := -fprofile-generate=$(PGO_DIR)
PGO_USE_CXXFLAGS := \
	-fprofile-use=$(PGO_DIR) \
	-fprofile-correction \
	-fno-peel-loops \
	-fno-tracer
PGO_USE_LDFLAGS := \
	-fprofile-use=$(PGO_DIR) \
	-fprofile-correction

LDLIBS :=

ifeq ($(OS),Windows_NT)
	EXEEXT := .exe

	fixpath = $(subst /,\,$(1))

	MKDIR_P = if not exist "$(call fixpath,$(1))" mkdir "$(call fixpath,$(1))"
	RM_F    = if exist "$(call fixpath,$(1))" del /q "$(call fixpath,$(1))"
	RMDIR_R = if exist "$(call fixpath,$(1))" rmdir /s /q "$(call fixpath,$(1))"
else
	EXEEXT :=
	MKDIR_P = mkdir -p "$(1)"
	RM_F    = rm -f "$(1)"
	RMDIR_R = rm -rf "$(1)"
endif

ifeq ($(BUILD_PROFILE),auto)
	OUTPUT_BASENAME := $(TARGET)
else
	OUTPUT_BASENAME := $(TARGET)-$(EFFECTIVE_PROFILE)
endif

DEFAULT_OUTPUT := $(BUILD_DIR)/$(OUTPUT_BASENAME)$(EXEEXT)

# External build systems may call make with EXE=<expected-binary-name>.
# When EXE is provided, write the final engine exactly there instead of
# always writing build/MagXTK-7SM2[.exe].
ifeq ($(strip $(EXE)),)
	OUTPUT := $(DEFAULT_OUTPUT)
	PGO_OUTPUT := $(BUILD_DIR)/$(TARGET)-$(EFFECTIVE_PROFILE)-pgo$(EXEEXT)
else
	OUTPUT := $(EXE)
	PGO_OUTPUT := $(EXE)
endif

PROFGEN_OUTPUT := $(BUILD_DIR)/$(TARGET)-$(EFFECTIVE_PROFILE)-profgen$(EXEEXT)
TIME_TEST_OUTPUT := $(BUILD_DIR)/TimeManagementTests-$(EFFECTIVE_PROFILE)$(EXEEXT)
SEARCH_REFERENCE_TEST_OUTPUT := $(BUILD_DIR)/SearchReferenceTests-$(EFFECTIVE_PROFILE)$(EXEEXT)

ifeq ($(PGO),1)
	AUTO_BUILD_TARGET := profile-build
	FINAL_OUTPUT := $(PGO_OUTPUT)
else
	AUTO_BUILD_TARGET := build
	FINAL_OUTPUT := $(OUTPUT)
endif

ifeq ($(OS),Windows_NT)
	RUN_CMD = .\$(call fixpath,$(FINAL_OUTPUT))
	RUN_PROFGEN_CMD = .\$(call fixpath,$(PROFGEN_OUTPUT))
	RUN_PGO_CMD = .\$(call fixpath,$(PGO_OUTPUT))
	PGO_BENCH_LOG_CMD = $(call fixpath,$(PGO_BENCH_LOG))
else
	RUN_CMD = ./$(FINAL_OUTPUT)
	RUN_PROFGEN_CMD = ./$(PROFGEN_OUTPUT)
	RUN_PGO_CMD = ./$(PGO_OUTPUT)
	PGO_BENCH_LOG_CMD = $(PGO_BENCH_LOG)
endif

.PHONY: all auto portable64 native64 avx2+bmi2 avx512+bmi2 variants cpu-variants detect build release releases \
	pgo profile-build pgo-step1 pgo-step2 pgo-step3 \
	pgo-generate-build pgo-use-build run time-test search-reference-test \
	clean objclean pgo-clean distclean

all: auto
auto: $(AUTO_BUILD_TARGET)

portable64:
	@$(MAKE) --no-print-directory BUILD_PROFILE=portable64 auto

native64:
	@$(MAKE) --no-print-directory BUILD_PROFILE=native64 auto

avx2+bmi2:
	@$(MAKE) --no-print-directory BUILD_PROFILE=avx2+bmi2 auto

avx512+bmi2:
	@$(MAKE) --no-print-directory BUILD_PROFILE=avx512+bmi2 auto

variants: portable64 native64

cpu-variants: avx2+bmi2 avx512+bmi2

detect:
	@echo Compiler target : $(TARGET_TRIPLE)
	@echo Requested profile: $(BUILD_PROFILE)
	@echo Selected profile : $(EFFECTIVE_PROFILE)
	@echo PGO enabled      : $(PGO)
	@echo CPU flags        : $(ARCH_FLAGS)
	@echo Output           : $(FINAL_OUTPUT)

build: $(OUTPUT)
release: auto
releases: variants

$(OUTPUT): $(OBJ)
	$(CXX) $(OBJ) -o $@ $(BASE_LDFLAGS) $(PGO_LDFLAGS) $(LDLIBS)

$(PROFGEN_OUTPUT): $(OBJ)
	$(CXX) $(OBJ) -o $@ $(BASE_LDFLAGS) $(PGO_LDFLAGS) $(LDLIBS)

$(PGO_OUTPUT): $(OBJ)
	$(CXX) $(OBJ) -o $@ $(BASE_LDFLAGS) $(PGO_LDFLAGS) $(LDLIBS)

$(OBJ): | $(MODULE_OBJ_DIRS)

$(OBJ_DIR)/%.o: %.cpp | $(OBJ_DIR)
	$(CXX) $(BASE_CXXFLAGS) $(PGO_CXXFLAGS) -c $< -o $@

$(FATHOM_OBJ): \
	$(FATHOM_DIR)/tbprobe.c \
	$(FATHOM_DIR)/tbprobe.h \
	$(FATHOM_DIR)/tbconfig.h \
	$(FATHOM_DIR)/tbchess.c \
	$(FATHOM_DIR)/stdendian.h | $(OBJ_DIR)
	$(CC) $(FATHOM_CFLAGS) -c $(FATHOM_DIR)/tbprobe.c -o $@

# Generate/copy/download embedded MNUE for clean clones.
ifeq ($(OS),Windows_NT)
build/$(MNUE_EMBEDDED_FILE):
	$(call MKDIR_P,$(BUILD_DIR))
	@echo Downloading embedded MNUE: $(MNUE_EMBEDDED_URL)
	curl.exe -L --fail --retry 5 --connect-timeout 20 "$(MNUE_EMBEDDED_URL)" -o "$(call fixpath,$@)"
else
build/$(MNUE_EMBEDDED_FILE):
	$(call MKDIR_P,$(BUILD_DIR))
	@if [ -f "$(MNUE_EMBEDDED_LOCAL)" ]; then \
		echo "Using local embedded MNUE: $(MNUE_EMBEDDED_LOCAL)"; \
		cp "$(MNUE_EMBEDDED_LOCAL)" "$@"; \
	else \
		echo "Downloading embedded MNUE: $(MNUE_EMBEDDED_URL)"; \
		curl -L --fail --retry 5 --connect-timeout 20 "$(MNUE_EMBEDDED_URL)" -o "$@"; \
	fi
endif

# === Embedded network (separate .S - no LTO, no PGO) ===
# The incbin source is a raw 20.1 MiB MNUE file. We compile it as a standalone
# assembly object so that -fprofile-generate, -flto, and whole-program analysis
# never touch this blob.
$(MNUE_EMBEDDED_OBJ): mnue/MnueEmbedded.S build/$(MNUE_EMBEDDED_FILE) Makefile | $(OBJ_DIR)/mnue
	$(CC) -x assembler-with-cpp -std=gnu99 -m64 -O2 \
		-DMNUE_EMBEDDED_PATH='"build/$(MNUE_EMBEDDED_FILE)"' \
		-I. \
		-c $< -o $@

$(OBJ_DIR)/mnue/Mnue.o: build/$(MNUE_EMBEDDED_FILE) Makefile

$(OBJ_DIR)/Uci.o: Makefile

$(OBJ_DIR):
	$(call MKDIR_P,$@)

$(MODULE_OBJ_DIRS): | $(OBJ_DIR)
	$(call MKDIR_P,$@)

$(PGO_DIR):
	$(call MKDIR_P,$@)

pgo-generate-build: PGO_CXXFLAGS = $(PGO_GENERATE_CXXFLAGS)
pgo-generate-build: PGO_LDFLAGS = $(PGO_GENERATE_LDFLAGS)
pgo-generate-build: $(PROFGEN_OUTPUT)

pgo-use-build: PGO_CXXFLAGS = $(PGO_USE_CXXFLAGS)
pgo-use-build: PGO_LDFLAGS = $(PGO_USE_LDFLAGS)
pgo-use-build: $(PGO_OUTPUT)

pgo: profile-build

profile-build:
	@echo Step 1/3. Building instrumented executable $(PROFGEN_OUTPUT) ...
	@$(MAKE) --no-print-directory pgo-step1
	@echo Step 2/3. Running bench to collect profile data ...
	@$(MAKE) --no-print-directory pgo-step2
	@echo Step 3/3. Rebuilding optimized executable with PGO $(PGO_OUTPUT) ...
	@$(MAKE) --no-print-directory pgo-step3

pgo-step1:
	@$(MAKE) --no-print-directory OBJ_VARIANT=pgo objclean
	@$(MAKE) --no-print-directory pgo-clean
	$(call RM_F,$(PROFGEN_OUTPUT))
	$(call RM_F,$(PGO_OUTPUT))
	@$(MAKE) --no-print-directory OBJ_VARIANT=pgo pgo-generate-build

pgo-step2: | $(PGO_DIR)
	@$(MAKE) --no-print-directory OBJ_VARIANT=pgo pgo-generate-build
	$(RUN_PROFGEN_CMD) $(PGO_BENCH_ARGS) > "$(PGO_BENCH_LOG_CMD)" 2>&1
	@echo PGO bench log written to $(PGO_BENCH_LOG)

pgo-step3:
	@$(MAKE) --no-print-directory OBJ_VARIANT=pgo objclean
	$(call RM_F,$(PGO_OUTPUT))
	@$(MAKE) --no-print-directory OBJ_VARIANT=pgo pgo-use-build

.PHONY: run
run: all
	$(RUN_CMD)

$(TIME_TEST_OUTPUT): tests/TimeManagementTests.cpp search/Time.cpp search/Time.h search/Search.h board/Position.h | $(OBJ_DIR)
	$(CXX) -std=c++20 $(ARCH_FLAGS) -O2 $(LOWLEVEL_FLAGS) $(ENCODING_FLAGS) $(WARN_FLAGS) \
		-iquote . tests/TimeManagementTests.cpp search/Time.cpp -o $@

time-test: $(TIME_TEST_OUTPUT)
	$(TIME_TEST_OUTPUT)

$(SEARCH_REFERENCE_TEST_OUTPUT): tests/SearchReferenceTests.cpp $(filter-out $(OBJ_DIR)/Main.o $(OBJ_DIR)/Uci.o,$(OBJ)) | $(OBJ_DIR)
	$(CXX) $(BASE_CXXFLAGS) tests/SearchReferenceTests.cpp \
		$(filter-out $(OBJ_DIR)/Main.o $(OBJ_DIR)/Uci.o,$(OBJ)) \
		-o $@ $(BASE_LDFLAGS) $(LDLIBS)

search-reference-test: $(SEARCH_REFERENCE_TEST_OUTPUT)
	$(SEARCH_REFERENCE_TEST_OUTPUT)

objclean:
	$(call RMDIR_R,$(OBJ_DIR))

pgo-clean:
	$(call RMDIR_R,$(PGO_DIR))

clean:
	$(call RMDIR_R,$(BUILD_DIR)/obj)
	$(call RM_F,$(BUILD_DIR)/$(TARGET)$(EXEEXT))
	$(call RM_F,$(BUILD_DIR)/$(TARGET)-portable64$(EXEEXT))
	$(call RM_F,$(BUILD_DIR)/$(TARGET)-native64$(EXEEXT))
	$(call RM_F,$(BUILD_DIR)/$(TARGET)-avx2+bmi2$(EXEEXT))
	$(call RM_F,$(BUILD_DIR)/$(TARGET)-avx512+bmi2$(EXEEXT))
	$(call RM_F,$(BUILD_DIR)/$(TARGET)-portable64-profgen$(EXEEXT))
	$(call RM_F,$(BUILD_DIR)/$(TARGET)-native64-profgen$(EXEEXT))
	$(call RM_F,$(BUILD_DIR)/$(TARGET)-avx2+bmi2-profgen$(EXEEXT))
	$(call RM_F,$(BUILD_DIR)/$(TARGET)-avx512+bmi2-profgen$(EXEEXT))
	$(call RM_F,$(BUILD_DIR)/$(TARGET)-portable64-pgo$(EXEEXT))
	$(call RM_F,$(BUILD_DIR)/$(TARGET)-native64-pgo$(EXEEXT))
	$(call RM_F,$(BUILD_DIR)/$(TARGET)-avx2+bmi2-pgo$(EXEEXT))
	$(call RM_F,$(BUILD_DIR)/$(TARGET)-avx512+bmi2-pgo$(EXEEXT))
	$(call RM_F,$(BUILD_DIR)/TimeManagementTests-portable64$(EXEEXT))
	$(call RM_F,$(BUILD_DIR)/TimeManagementTests-native64$(EXEEXT))

distclean: clean
	$(call RMDIR_R,$(BUILD_DIR)/pgo)

-include $(DEP)
