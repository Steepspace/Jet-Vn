#!/usr/bin/env bash
export USER="$(id -u -n)"
export LOGNAME=${USER}
export HOME=/sphenix/u/${LOGNAME}

# sPHENIX build comes from the submit file environment, set by condor_utils
build="${SPHENIX_BUILD:-${OFFLINE_MAIN:+$(basename "$OFFLINE_MAIN")}}"
if [[ -z "$build" ]]; then
    echo "Error: SPHENIX_BUILD is not set" >&2
    exit 1
fi

source /opt/sphenix/core/bin/sphenix_setup.sh -n "$build"

f4a_macro=${1}
input_list=${2}
nEvents=${3}
dbtag=${4}
logInterval=${5}
submitDir=${6}

if [[ -n "$_CONDOR_SCRATCH_DIR" && -d "$_CONDOR_SCRATCH_DIR" ]]
then
    cd "$_CONDOR_SCRATCH_DIR" || { echo "Failed to cd to $_CONDOR_SCRATCH_DIR" >&2; exit 1; }

    # Check if the list file exists in the scratch dir
    if [[ ! -f "$input_list" ]]; then
        echo "Error: Input list file '$input_list' not found!" >&2
        exit 1
    fi

    # print the environment - needed for debugging
    printenv

    echo "Fetching all DST files at once from: $input_list"

    # Transfer all files in the list upfront: list lines are calofit,zdc,sepd
    # DST_CALOFITTING is needed for MbdReco (MBD t0 for the ZdcReco satellite correction)
    cut -d ',' -f 1,2 "$input_list" | tr ',' '\n' > dst_calofit_zdc.list

    getinputfiles.pl --verbose --filelist dst_calofit_zdc.list || {
        echo "Error: getinputfiles.pl failed for dst_calofit_zdc.list at $(date) on $(hostname)" >&2
        mkdir -p "$submitDir/failures"
        echo "getinputfiles failure (dst_calofit_zdc) for $input_list on $(hostname) at $(date)" >> "$submitDir/failures/failure-log.txt"
        exit 1
    }

    ls -lah

    # Loop over each line (calofit,zdc DST pair) in the input list to process them
    while IFS=',' read -r current_calofit current_zdc _ || [[ -n "$current_calofit" ]]; do
        # Skip empty lines
        if [[ -z "$current_calofit" || -z "$current_zdc" ]]; then continue; fi

        # Ensure we are just using the filenames in case the list had full paths
        local_calofit=$(basename "$current_calofit")
        local_file=$(basename "$current_zdc")

        echo "--------------------------------------------------------"
        echo "Processing: $local_calofit + $local_file"

        # Run the macro on the current local DSTs
        root -b -l -q "$f4a_macro(\"$local_calofit\", \"$local_file\", \"output\", $nEvents, \"$dbtag\", $logInterval)"

        root_exit=$?
        if [ $root_exit -ne 0 ]; then
            echo "Error: ROOT macro failed with exit code $root_exit at $(date) on $(hostname)! Aborting transfer." >&2
            mkdir -p "$submitDir/failures"
            echo "ROOT failure (exit code $root_exit) for $input_list on $(hostname) at $(date)" >> "$submitDir/failures/failure-log.txt"
            exit $root_exit
        fi

    done < "$input_list"

else
    echo "condor scratch NOT set" >&2
    exit 1
fi

echo "All Done and Transferring Files Back at $(date)"

# Define maximum retries and a counter
max_retries=5
count=0
success=0

while [ $count -lt $max_retries ]; do
    if cp -rv output/. "$submitDir"; then
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
    echo "CP transfer failure for $input_list on $(hostname) at $(date)" >> "$submitDir/failures/failure-log.txt"
    exit 1
fi

echo "Finished successfully at $(date)"
