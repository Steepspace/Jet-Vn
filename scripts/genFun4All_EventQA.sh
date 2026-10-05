#!/usr/bin/env bash
export USER="$(id -u -n)"
export LOGNAME=${USER}
export HOME=/sphenix/u/${LOGNAME}
export MYINSTALL="$HOME/Documents/sPHENIX/install"

source /opt/sphenix/core/bin/sphenix_setup.sh -n new
source /opt/sphenix/core/bin/setup_local.sh $MYINSTALL

f4a_macro=${1}
input=${2}
output=${3}
nEvents=${4}
dbtag=${5}
submitDir=${6}

# extract runnumber from file name
file=$(basename "$input")
IFS='-' read -r p1 p2 p3 <<< "$file"
run=$(echo "$p2" | sed 's/^0*//') # Remove leading zeros using sed

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

    ls -lah
else
    echo "condor scratch NOT set" >&2
    exit 1
fi

# print the environment - needed for debugging
printenv

mkdir -p "$run"

echo "Starting ROOT macro at $(date) on $(hostname)"
root -b -l -q "$f4a_macro(\"dst_calofit.list\", \"dst_zdc.list\", \"dst_sepd.list\", \"$run/$output\", $nEvents, \"$dbtag\")"

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
