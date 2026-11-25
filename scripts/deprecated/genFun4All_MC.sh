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
    getinputfiles.pl --verbose --filelist "$input"
    ls -lah
else
    echo "condor scratch NOT set" >&2
    exit 1
fi

# print the environment - needed for debugging
printenv

mkdir -p "$run"

root -b -l -q "$f4a_macro($nEvents, \"$input\", \"$run/$output\", \"$dbtag\")"

echo "All Done and Transferring Files Back"
cp -rv "$run" "$submitDir"

echo "Finished"
