#!/usr/bin/env bash
export USER="$(id -u -n)"
export LOGNAME=${USER}
export HOME=/sphenix/u/${LOGNAME}
if [ -n "$11" ]; then
    build="$11"
elif [ -n "$OFFLINE_MAIN" ]; then
    build="$(basename "$OFFLINE_MAIN")"
else
    build="new"
fi
myinstall_arg=${12:-${MYINSTALL:-default}}

if [[ -z "$myinstall_arg" || "$myinstall_arg" == "default" ]]; then
    export MYINSTALL="$HOME/Documents/sPHENIX/install"
elif [[ "$myinstall_arg" == "none" ]]; then
    export MYINSTALL=""
else
    export MYINSTALL="$myinstall_arg"
fi

source /opt/sphenix/core/bin/sphenix_setup.sh -n "$build"
if [[ -n "$MYINSTALL" && -d "$MYINSTALL" ]]; then
    source /opt/sphenix/core/bin/setup_local.sh "$MYINSTALL"
fi

jetAna_bin=${1}
input=${2}
input_calo_mbd=${3}
jet_pt_min=${4}
submitDir=${5}
do_iter=${6:-1}
do_mult=${7:-1}
do_unsub=${8:-1}
do_rcone=${9:-1}
lead_jet_pt_threshold=${10:-100}

# extract runnumber from file name
file=$(basename "$input")
run=$(head -n 1 "$input" | grep -oP '(?<=/)\d+(?=/tree/)' || head -n 1 "$input" | grep -oP '\d{5,8}')
if [ -z "$run" ]; then
    echo "Error: Could not extract run number from $input at $(date) on $(hostname)!" >&2
    mkdir -p "$submitDir/failures"
    echo "run number extraction failure for $file on $(hostname) at $(date)" >> "$submitDir/failures/failure-log.txt"
    exit 1
fi
input_file="$file"
input_calo_mbd_file=$(basename "$input_calo_mbd")

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

    if [ -n "$input_calo_mbd" ]; then
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

mkdir -p "$run"

echo "Starting Jet-Anav3 binary at $(date) on $(hostname)"
$jetAna_bin "$input_file" "$input_calo_mbd_file" 0 "$jet_pt_min" "$run" 0 "$do_iter" "$do_mult" "$do_unsub" "$do_rcone" "$lead_jet_pt_threshold"

bin_exit=$?
if [ $bin_exit -ne 0 ]; then
    echo "Error: Jet-Anav3 binary failed with exit code $bin_exit at $(date) on $(hostname)! Aborting transfer." >&2
    mkdir -p "$submitDir/failures"
    echo "Jet-Anav3 failure (exit code $bin_exit) for $file on $(hostname) at $(date)" >> "$submitDir/failures/failure-log.txt"
    exit $bin_exit
fi

echo "All Done and Transferring Files Back at $(date)"

# Define maximum retries and a counter
max_retries=5
count=0
success=0

while [ $count -lt $max_retries ]; do
    if cp -rv "$run" "$submitDir"; then
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
