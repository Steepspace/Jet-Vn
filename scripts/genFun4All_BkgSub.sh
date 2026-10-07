#!/usr/bin/env bash
export USER="$(id -u -n)"
export LOGNAME=${USER}
export HOME=/sphenix/u/${LOGNAME}
if [ -n "$16" ]; then
    build="$16"
elif [ -n "$OFFLINE_MAIN" ]; then
    build="$(basename "$OFFLINE_MAIN")"
else
    build="new"
fi
myinstall_arg=${17:-${MYINSTALL:-default}}

if [[ -z "$myinstall_arg" || "$myinstall_arg" == "default" ]]; then
    export MYINSTALL="$HOME/Documents/sPHENIX/install"
elif [[ "$myinstall_arg" == "none" ]]; then
    export MYINSTALL=""
else
    export MYINSTALL="$myinstall_arg"
fi

source /opt/sphenix/core/bin/sphenix_setup.sh -n "$build"
if [[ -n "$MYINSTALL" ]]; then
    source /opt/sphenix/core/bin/setup_local.sh "$MYINSTALL"
fi

f4a_macro=${1}
input=${2}
input_calib=${3}
output=${4}
output_tree=${5}
nEvents=${6}
dbtag=${7}
do_flow=${8}
eta_calib_path=${9}
event_list_path=${10}
do_rcone=${11:-0}
do_mult=${12:-1}
do_neg_energy_threshold=${13:-1}
neg_energy_threshold=${14:--2.0}
submitDir=${15}

# extract runnumber from file name
file=$(basename "$input")
IFS='-' read -r p1 p2 p3 <<< "$file"
run=$(echo "$p2" | sed 's/^0*//') # Remove leading zeros using sed

# Check if input_calib is a path or a keyword
if [[ "$input_calib" == "default" ]]; then
    calib_file="default"
else
    calib_file=$(basename "$input_calib")
fi

# Check if eta_calib_path is a path or a keyword / empty
if [[ -z "$eta_calib_path" || "$eta_calib_path" == "none" || "$eta_calib_path" == "default" ]]; then
    eta_calib_file=""
else
    eta_calib_file=$(basename "$eta_calib_path")
fi

# Check if event_list_path is a path or a keyword / empty
if [[ -z "$event_list_path" || "$event_list_path" == "none" || "$event_list_path" == "default" ]]; then
    event_list_file=""
else
    event_list_file=$(basename "$event_list_path")
fi

if [[ -n "$_CONDOR_SCRATCH_DIR" && -d "$_CONDOR_SCRATCH_DIR" ]]
then
    cd "$_CONDOR_SCRATCH_DIR" || { echo "Failed to cd to $_CONDOR_SCRATCH_DIR" >&2; exit 1; }

    echo "Reading inputs from: $input"

    # Initialize empty list files for Fun4All
    > dst_calofit.list
    > dst_zdc.list
    > dst_sepd.list

    # Ensure failure log directory exists
    mkdir -p "$submitDir/failures"

    total_segments=0
    successful_segments=0

    while IFS= read -r line || [ -n "$line" ]; do
        # Skip empty lines
        [ -z "$line" ] && continue

        total_segments=$((total_segments + 1))

        # Extract comma-separated paths for this segment
        IFS=',' read -r calofit zdc sepd <<< "$line"
        calofit=$(echo "$calofit" | tr -d '[:space:]')
        zdc=$(echo "$zdc" | tr -d '[:space:]')
        sepd=$(echo "$sepd" | tr -d '[:space:]')

        [ -z "$calofit" ] && continue

        segment_ok=1

        # 1. Fetch calofitting file via getinputfiles.pl
        if ! getinputfiles.pl --verbose "$calofit"; then
            echo "Error: getinputfiles.pl failed for $calofit in $file at $(date) on $(hostname)" >&2
            echo "getinputfiles failure (dst_calofit) for $calofit in $file on $(hostname) at $(date)" >> "$submitDir/failures/failure-log.txt"
            segment_ok=0
        fi

        # 2. Copy ZDC file
        if [ $segment_ok -eq 1 ]; then
            if ! cp -v "$zdc" .; then
                echo "Error: Failed to copy ZDC file $zdc in $file at $(date) on $(hostname)" >&2
                echo "copy failure (dst_zdc) for $zdc in $file on $(hostname) at $(date)" >> "$submitDir/failures/failure-log.txt"
                rm -f "$(basename "$calofit")"
                segment_ok=0
            fi
        fi

        # 3. Copy sEPD file
        if [ $segment_ok -eq 1 ]; then
            if ! cp -v "$sepd" .; then
                echo "Error: Failed to copy sEPD file $sepd in $file at $(date) on $(hostname)" >&2
                echo "copy failure (dst_sepd) for $sepd in $file on $(hostname) at $(date)" >> "$submitDir/failures/failure-log.txt"
                rm -f "$(basename "$calofit")"
                rm -f "$(basename "$zdc")"
                segment_ok=0
            fi
        fi

        # If all 3 files succeeded, register this segment
        if [ $segment_ok -eq 1 ]; then
            basename "$calofit" >> dst_calofit.list
            basename "$zdc" >> dst_zdc.list
            basename "$sepd" >> dst_sepd.list
            successful_segments=$((successful_segments + 1))
        fi
    done < "$input"

    echo "Fetched $successful_segments of $total_segments segments successfully."

    # If no segments could be fetched, abort this job
    if [ ! -s dst_calofit.list ]; then
        echo "Aborted: All segments failed to fetch (missing catalog/data files) for $file at $(date) on $(hostname)"
        echo "Error: All segments failed to fetch for $file at $(date) on $(hostname)! Aborting." >&2
        echo "all segments failed for $file on $(hostname) at $(date)" >> "$submitDir/failures/failure-log.txt"
        exit 1
    fi

    test -e "$input_calib" && cp -v "$input_calib" .
    if [[ -n "$eta_calib_path" && "$eta_calib_path" != "none" && "$eta_calib_path" != "default" ]]; then
        test -e "$eta_calib_path" && cp -v "$eta_calib_path" .
    fi
    if [[ -n "$event_list_path" && "$event_list_path" != "none" && "$event_list_path" != "default" ]]; then
        test -e "$event_list_path" && cp -v "$event_list_path" .
    fi
    ls -lah
else
    echo "condor scratch NOT set" >&2
    exit 1
fi

# print the environment - needed for debugging
printenv

mkdir -p "$run/hist" "$run/tree"

echo "Starting ROOT macro at $(date) on $(hostname)"
root -b -l -q "$f4a_macro(\"dst_calofit.list\", \"dst_zdc.list\", \"dst_sepd.list\", \"$calib_file\", \"$run/hist/$output\", \"$run/tree/$output_tree\", $do_flow, $nEvents, 0, 0, \"$dbtag\", \"$eta_calib_file\", \"$event_list_file\", $do_rcone, $do_mult, \"\", $do_neg_energy_threshold, $neg_energy_threshold)"

root_exit=$?
if [ $root_exit -ne 0 ]; then
    echo "Error: ROOT macro failed with exit code $root_exit at $(date) on $(hostname)! Aborting transfer." >&2
    mkdir -p "$submitDir/failures"
    echo "ROOT failure (exit code $root_exit) for $file on $(hostname) at $(date)" >> "$submitDir/failures/failure-log.txt"
    exit $root_exit
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
