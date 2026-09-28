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
do_sepd=${7:-auto}

# Determine whether sEPD should be processed
use_sepd=0
if [[ "$do_sepd" == "1" || "$do_sepd" == "true" || "$do_sepd" == "True" ]]; then
    use_sepd=1
elif [[ "$do_sepd" == "0" || "$do_sepd" == "false" || "$do_sepd" == "False" ]]; then
    use_sepd=0
else
    # Auto-detect: check if input list has a non-empty 3rd comma-separated field
    first_line=$(head -n 1 "$input")
    f3=$(echo "$first_line" | cut -d ',' -f 3)
    if [[ -n "$f3" && "$f3" != "$first_line" ]]; then
        use_sepd=1
    fi
fi

# extract runnumber from file name
file=$(basename "$input")
IFS='-' read -r p1 p2 p3 <<< "$file"
run=$(echo "$p2" | sed 's/^0*//') # Remove leading zeros using sed

if [[ -n "$_CONDOR_SCRATCH_DIR" && -d "$_CONDOR_SCRATCH_DIR" ]]
then
    cd "$_CONDOR_SCRATCH_DIR" || { echo "Failed to cd to $_CONDOR_SCRATCH_DIR" >&2; exit 1; }

    echo "Reading inputs from: $input"

    cut -d ',' -f 1 "$input" > dst_calofit.list
    cut -d ',' -f 2 "$input" > dst_zdc.list

    getinputfiles.pl --verbose --filelist dst_calofit.list || {
        echo "Error: getinputfiles.pl failed for dst_calofit.list at $(date) on $(hostname)" >&2
        mkdir -p "$submitDir/failures"
        echo "getinputfiles failure (dst_calofit) for $file on $(hostname) at $(date)" >> "$submitDir/failures/failure-log.txt"
        exit 1
    }

    if [[ $use_sepd -eq 1 ]]; then
        cut -d ',' -f 3 "$input" > dst_sepd.list
        getinputfiles.pl --verbose --filelist dst_sepd.list || {
            echo "Error: getinputfiles.pl failed for dst_sepd.list at $(date) on $(hostname)" >&2
            mkdir -p "$submitDir/failures"
            echo "getinputfiles failure (dst_sepd) for $file on $(hostname) at $(date)" >> "$submitDir/failures/failure-log.txt"
            exit 1
        }
    fi

    # Create/clear a temporary file for the basenames
    > dst_zdc_local.list

    while IFS= read -r file; do
        # Skip empty lines if there are any
        [ -z "$file" ] && continue

        # Copy the file to the current directory
        cp -v "$file" . || {
            echo "Error: Failed to copy ZDC file $file at $(date) on $(hostname)" >&2
            mkdir -p "$submitDir/failures"
            echo "copy failure (dst_zdc) for $file on $(hostname) at $(date)" >> "$submitDir/failures/failure-log.txt"
            exit 1
        }

        # Extract just the filename and save it to our local list
        basename "$file" >> dst_zdc_local.list
    done < dst_zdc.list

    # Overwrite the original list with the basename-only list
    mv dst_zdc_local.list dst_zdc.list

    ls -lah
else
    echo "condor scratch NOT set" >&2
    exit 1
fi

# print the environment - needed for debugging
printenv

mkdir -p "$run"

echo "Starting ROOT macro at $(date) on $(hostname)"
if [[ $use_sepd -eq 1 ]]; then
    echo "Running Event QA with sEPD enabled"
    root -b -l -q "$f4a_macro(\"dst_calofit.list\", \"dst_zdc.list\", \"dst_sepd.list\", \"$run/$output\", $nEvents, \"$dbtag\")"
else
    echo "Running Event QA without sEPD"
    root -b -l -q "$f4a_macro(\"dst_calofit.list\", \"dst_zdc.list\", \"$run/$output\", $nEvents, \"$dbtag\")"
fi

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
