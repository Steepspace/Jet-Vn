#!/usr/bin/env bash

# Usage: ./builds.sh <project> [project ...] [-f] [-i install_dir] [-b build_dir]
# Examples:
#   ./builds.sh jetbackground
#   ./builds.sh jetbackground -f
#   ./builds.sh jetbase jetbackground
#   ./builds.sh jetbase jetbackground -f -b custom_builds
#   ./builds.sh jetbackground -i /path/to/install
#
# Projects are built in the order given; the script stops at the first failure.
# Install prefix defaults to $MYINSTALL; -i overrides it.
# -i only affects the configure step, so it has no effect with -f.
# Build directory defaults to $BUILD_BASE_DIR, or "builds" if unset.

usage() {
    echo "Usage: $0 <source_path_or_name> [source_path_or_name ...] [-f] [-i install_dir] [-b build_dir]"
}

PROJECTS=()
FAST_MODE=""
BUILD_BASE="${BUILD_BASE_DIR:-builds}"
INSTALL_PREFIX="${MYINSTALL:-}"

# Parse arguments: -f (fast mode), -i <install_dir>, -b <build_dir>, and project names/paths
while [ $# -gt 0 ]; do
    case "$1" in
        -f)
            FAST_MODE="-f"
            ;;
        -i)
            if [ -z "$2" ]; then
                echo "Error: -i requires an install directory."
                exit 1
            fi
            INSTALL_PREFIX="$2"
            shift
            ;;
        -b)
            if [ -z "$2" ]; then
                echo "Error: -b requires a build directory."
                exit 1
            fi
            BUILD_BASE="$2"
            shift
            ;;
        -*)
            echo "Error: Unknown option '$1'."
            usage
            exit 1
            ;;
        *)
            PROJECTS+=("$1")
            ;;
    esac
    shift
done

# Check if at least one project name/path was provided
if [ ${#PROJECTS[@]} -eq 0 ]; then
    echo "Error: Please provide at least one project name or path."
    usage
    exit 1
fi

# SRC_DIR env override only makes sense for a single project
if [ -n "$SRC_DIR" ] && [ ${#PROJECTS[@]} -gt 1 ]; then
    echo "Error: SRC_DIR is set but multiple projects were given."
    exit 1
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

# Ensure BUILD_BASE is resolved relative to execution directory if relative
if [[ "$BUILD_BASE" != /* ]]; then
    BUILD_BASE="$(pwd)/$BUILD_BASE"
fi

# Build a single project. Runs in a subshell so cd and exit stay local.
build_project() (
    SRC_INPUT="$1"
    PROJ_NAME="$(basename "$SRC_INPUT")"

    # Resolve Source Directory (SRC_DIR)
    if [ -n "$SRC_DIR" ]; then
        : # Use explicit environment variable if set
    elif [ -d "$SRC_INPUT" ]; then
        SRC_DIR="$( cd "$SRC_INPUT" && pwd )"
    elif [ -d "$ROOT_DIR/$SRC_INPUT" ]; then
        SRC_DIR="$( cd "$ROOT_DIR/$SRC_INPUT" && pwd )"
    else
        SRC_DIR="$ROOT_DIR/$SRC_INPUT"
    fi

    BUILD_DIR="$BUILD_BASE/$PROJ_NAME"

    # --- Logic Branching ---

    if [ "$FAST_MODE" == "-f" ]; then
        echo ">>> Fast Mode: Running 'make install' for $PROJ_NAME in $BUILD_DIR..."
        make install -j8 --directory "$BUILD_DIR" || exit 1
    else
        echo ">>> Full Build: Refreshing and configuring $PROJ_NAME..."
        echo "    Source: $SRC_DIR"
        echo "    Build:  $BUILD_DIR"
        echo "    Install: $INSTALL_PREFIX"

        # 1. Clean and Create Build Dir
        rm -rf "$BUILD_DIR" && mkdir -p "$BUILD_DIR" || exit 1

        # 2. Configure (autogen)
        cd "$BUILD_DIR" && "$SRC_DIR"/autogen.sh --prefix="$INSTALL_PREFIX" || exit 1

        # 3. Build and Install
        if command -v bear >/dev/null 2>&1; then
            echo ">>> Generating compilation database with bear..."
            bear -- make -j8 || exit 1
            make install -j8 || exit 1

            # 4. Link compile_commands.json to source directory
            if [ -f "$BUILD_DIR/compile_commands.json" ]; then
                echo ">>> Linking compile_commands.json to $SRC_DIR..."
                ln -sfn "$BUILD_DIR/compile_commands.json" "$SRC_DIR/compile_commands.json"
            fi
        else
            echo ">>> 'bear' not found, running standard build..."
            make install -j8 || exit 1
        fi
    fi
)

DONE=()
for proj in "${PROJECTS[@]}"; do
    if ! build_project "$proj"; then
        echo ">>> FAILED: $proj"
        [ ${#DONE[@]} -gt 0 ] && echo ">>> Completed before failure: ${DONE[*]}"
        exit 1
    fi
    DONE+=("$proj")
done

if [ ${#PROJECTS[@]} -gt 1 ]; then
    echo ">>> All projects built: ${DONE[*]}"
fi
