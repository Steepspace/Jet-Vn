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

# p+p MC JES (Fun4All_PP_JES.C): one chunk of aligned DST_CALO_CLUSTER / DST_GLOBAL / DST_TRUTH_JET per job
f4a_macro=${1}
list_calo=${2}   # lists of LFNs, line i of each list is the same segment
list_global=${3}
list_truth=${4}
sample=${5}      # pT-hat sample (e.g. jet12)
output=${6}
output_tree=${7}
nEvents=${8}
dbtag=${9}
jet_pt_min=${10}
truth_jet_pt_min=${11}
submitDir=${12}

# outputs go to <sample>/{hist,tree}, so the pT-hat samples stay separate
outdir="$sample"

if [[ -n "$_CONDOR_SCRATCH_DIR" && -d "$_CONDOR_SCRATCH_DIR" ]]
then
    cd "$_CONDOR_SCRATCH_DIR" || { echo "Failed to cd to $_CONDOR_SCRATCH_DIR" >&2; exit 1; }

    # Ensure failure log directory exists
    mkdir -p "$submitDir/failures"

    for list in "$list_calo" "$list_global" "$list_truth"; do
        if ! getinputfiles.pl --verbose --filelist "$list"; then
            echo "Error: Failed to fetch the files of $list at $(date) on $(hostname)" >&2
            echo "getinputfiles failure for $list on $(hostname) at $(date)" >> "$submitDir/failures/failure-log.txt"
            exit 1
        fi
        # the lists hold bare file names, which now resolve in the scratch directory
        cp -v "$list" .
    done
    ls -lah
else
    echo "condor scratch NOT set" >&2
    exit 1
fi

# print the environment - needed for debugging
printenv

mkdir -p "$outdir/hist" "$outdir/tree"

echo "Starting ROOT macro at $(date) on $(hostname)"
root -b -l -q "$f4a_macro(\"$(basename "$list_calo")\", \"$(basename "$list_global")\", \"$(basename "$list_truth")\", \"$outdir/hist/$output\", \"$outdir/tree/$output_tree\", $nEvents, 0, \"$dbtag\", $jet_pt_min, $truth_jet_pt_min)"

root_exit=$?
if [ $root_exit -ne 0 ]; then
    echo "Error: ROOT macro failed with exit code $root_exit at $(date) on $(hostname)! Aborting transfer." >&2
    mkdir -p "$submitDir/failures"
    echo "ROOT failure (exit code $root_exit) for $list_calo on $(hostname) at $(date)" >> "$submitDir/failures/failure-log.txt"
    exit $root_exit
fi

echo "All Done and Transferring Files Back at $(date)"

# Define maximum retries and a counter
max_retries=5
count=0
success=0

while [ $count -lt $max_retries ]; do
    if cp -rv "$outdir" "$submitDir"; then
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
    echo "CP transfer failure for $list_calo on $(hostname) at $(date)" >> "$submitDir/failures/failure-log.txt"
    exit 1
fi

echo "Finished successfully at $(date)"
