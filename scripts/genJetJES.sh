#!/usr/bin/env bash
export USER="$(id -u -n)"
export LOGNAME=${USER}
export HOME=/sphenix/u/${LOGNAME}
# sPHENIX build (and local install) come from the submit file environment, set by condor_utils
build="${SPHENIX_BUILD:-${OFFLINE_MAIN:+$(basename "$OFFLINE_MAIN")}}"
if [[ -z "$build" ]]; then
    echo "Error: SPHENIX_BUILD is not set" >&2
    exit 1
fi
myinstall="${SPHENIX_MYINSTALL:-${MYINSTALL:-none}}"

source /opt/sphenix/core/bin/sphenix_setup.sh -n "$build"
if [[ "$myinstall" != "none" ]]; then
    export MYINSTALL="$myinstall"
    source /opt/sphenix/core/bin/setup_local.sh "$MYINSTALL"
fi

jetJES_bin=${1}
input=${2}
input_calo_mbd=${3}
submitDir=${4}
lead_truth_pt_min=${5}
lead_truth_pt_max=${6}
sigma_cut=${7:-3.5}
# overlay pT-hat sample (e.g. Jet20): output directory, so hadd merges each sample separately
sample=${8:-}
# 1: p+p MC trees (Fun4All_PP_JES), no Calo-MBD file
pp=${9:-0}

# extract runnumber from file name (p+p MC: the sample is the output directory, no run number needed)
file=$(basename "$input")
run=$(head -n 1 "$input" | grep -oP '(?<=/)\d+(?=/tree/)' || head -n 1 "$input" | grep -oP '\d{5,8}')
if [[ "$pp" == "1" ]]; then
    run=${run:-pp}
fi
if [ -z "$run" ]; then
    echo "Error: Could not extract run number from $input at $(date) on $(hostname)!" >&2
    mkdir -p "$submitDir/failures"
    echo "run number extraction failure for $file on $(hostname) at $(date)" >> "$submitDir/failures/failure-log.txt"
    exit 1
fi
input_file="$file"
if [[ "$pp" == "1" ]]; then
    input_calo_mbd_file="none"
else
    input_calo_mbd_file=$(basename "$input_calo_mbd")
fi

if [[ -n "$_CONDOR_SCRATCH_DIR" && -d "$_CONDOR_SCRATCH_DIR" ]]
then
    cd "$_CONDOR_SCRATCH_DIR" || { echo "Failed to cd to $_CONDOR_SCRATCH_DIR" >&2; exit 1; }
    mkdir input

    # Ensure failure log directory exists
    mkdir -p "$submitDir/failures"

    echo "Reading inputs from list: $input at $(date) on $(hostname)"
    cat "$input" | xargs -I {} -P 4 cp -v {} input/

    realpath input/* > "$input_file"

    if [ ! -s "$input_file" ]; then
        echo "Error: All input files failed to copy for $file at $(date) on $(hostname)! Aborting." >&2
        echo "copy failure (all input files) for $file on $(hostname) at $(date)" >> "$submitDir/failures/failure-log.txt"
        exit 1
    fi

    if [[ "$pp" == "1" ]]; then
        echo "p+p MC mode: no Calo-MBD file"
    elif [ -n "$input_calo_mbd" ]; then
        if ! cp -v "$input_calo_mbd" .; then
            echo "Error: Failed to copy Calo-MBD file $input_calo_mbd in $file at $(date) on $(hostname)" >&2
            echo "copy failure (input_calo_mbd) for $input_calo_mbd in $file on $(hostname) at $(date)" >> "$submitDir/failures/failure-log.txt"
            exit 1
        fi
    else
        echo "Error: input_calo_mbd was not provided for $file at $(date) on $(hostname)! Aborting." >&2
        echo "missing input_calo_mbd for $file on $(hostname) at $(date)" >> "$submitDir/failures/failure-log.txt"
        exit 1
    fi

    ls -lah
else
    echo "condor scratch NOT set at $(date) on $(hostname)" >&2
    exit 1
fi

# print the environment - needed for debugging
printenv

out_dir="${sample:-$run}"
mkdir -p "$out_dir"

echo "Starting Jet-JES binary at $(date) on $(hostname)"
$jetJES_bin "$input_file" "$input_calo_mbd_file" 0 "$out_dir" "$lead_truth_pt_min" "$lead_truth_pt_max" "$sigma_cut" 0 "$pp"

bin_exit=$?
if [ $bin_exit -ne 0 ]; then
    echo "Error: Jet-JES binary failed with exit code $bin_exit at $(date) on $(hostname)! Aborting transfer." >&2
    mkdir -p "$submitDir/failures"
    echo "Jet-JES failure (exit code $bin_exit) for $file on $(hostname) at $(date)" >> "$submitDir/failures/failure-log.txt"
    exit $bin_exit
fi

echo "All Done and Transferring Files Back at $(date)"

# Define maximum retries and a counter
max_retries=5
count=0
success=0

while [ $count -lt $max_retries ]; do
    if cp -rv "$out_dir" "$submitDir"; then
        success=1
        break
    else
        count=$((count + 1))
        echo "cp failed (likely GPFS lag). Retrying ($count/$max_retries) in 15 seconds..." >&2
        sleep 15
    fi
done

if [ $success -eq 0 ]; then
    echo "Error: cp failed permanently after $max_retries attempts at $(date)." >&2
    mkdir -p "$submitDir/failures"
    echo "CP transfer failure for $file on $(hostname) at $(date)" >> "$submitDir/failures/failure-log.txt"
    exit 1
fi

echo "Finished successfully at $(date)"
