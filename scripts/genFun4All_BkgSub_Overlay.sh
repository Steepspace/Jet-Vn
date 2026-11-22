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

# Data+MC overlay mode of Fun4All_BkgSub.C: one overlay job directory (OutDir) per job
f4a_macro=${1}
input_global=${2}
input_jet=${3}
input_g4hit=${4} # not needed for the trees, kept so the list columns map one to one
input_calo=${5}
input_zdc=${6}  # data ZDC DST of the same run and segment
input_sepd=${7} # data sEPD DST of the same run and segment
input_calib=${8}
output=${9}
output_tree=${10}
nEvents=${11}
dbtag=${12}
do_flow=${13}
eta_calib_path=${14}
do_rcone=${15:-0}
do_neg_energy_threshold=${16:-1}
neg_energy_threshold=${17:--2.0}
zs_calib_cemc=${18:-none}
zs_calib_hcalin=${19:-none}
zs_calib_hcalout=${20:-none}
do_emfrac_calib=${21:-0}
jet_pt_min=${22:-10}
truth_jet_pt_min=${23:-5}
submitDir=${24}
input_check_macro=${25} # macros/check_overlay_inputs.C: checks the input copies before the f4a macro

# the data-only DST_CALOFITTING (event aligned with the overlay DSTs) sits next to them
input_calofit=$(ls "$(dirname "$input_global")"/DST_CALOFITTING-*.root 2>/dev/null)
if [[ $(echo "$input_calofit" | grep -c .) -ne 1 ]]; then
    echo "Error: expected one DST_CALOFITTING next to $input_global, found: $input_calofit" >&2
    mkdir -p "$submitDir/failures"
    echo "DST_CALOFITTING lookup failure for $input_global on $(hostname) at $(date)" >> "$submitDir/failures/failure-log.txt"
    exit 1
fi

file_calofit=$(basename "$input_calofit")
file_jet=$(basename "$input_jet")
file_calo=$(basename "$input_calo")
file_zdc=$(basename "$input_zdc")
file_sepd=$(basename "$input_sepd")

# extract sample and runnumber from file name (DST_CALO_Jet12-00000021-data-00068144-00000.root)
IFS='-' read -r p1 p2 p3 p4 p5 <<< "$file_calo"
run=$(echo "$p4" | sed 's/^0*//') # Remove leading zeros using sed
sample=${p1#DST_CALO_}
# outputs go to <sample>/<run>/{hist,tree}, so the pT-hat samples stay separate
outdir="$sample/$run"

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

# ZS cross calib T0 files: "none" means use the CDB
zs_calib_files=()
for zs_calib_path in "$zs_calib_cemc" "$zs_calib_hcalin" "$zs_calib_hcalout"; do
    if [[ -z "$zs_calib_path" || "$zs_calib_path" == "none" || "$zs_calib_path" == "default" ]]; then
        zs_calib_files+=("none")
    else
        zs_calib_files+=("$(basename "$zs_calib_path")")
    fi
done

if [[ -n "$_CONDOR_SCRATCH_DIR" && -d "$_CONDOR_SCRATCH_DIR" ]]
then
    cd "$_CONDOR_SCRATCH_DIR" || { echo "Failed to cd to $_CONDOR_SCRATCH_DIR" >&2; exit 1; }

    # Ensure failure log directory exists
    mkdir -p "$submitDir/failures"

    for dst in "$input_calofit" "$input_zdc" "$input_sepd" "$input_calo" "$input_jet"; do
        if ! cp -v "$dst" .; then
            echo "Error: Failed to copy $dst at $(date) on $(hostname)" >&2
            echo "copy failure for $dst on $(hostname) at $(date)" >> "$submitDir/failures/failure-log.txt"
            exit 1
        fi
    done

    test -e "$input_calib" && cp -v "$input_calib" .
    if [[ -n "$eta_calib_path" && "$eta_calib_path" != "none" && "$eta_calib_path" != "default" ]]; then
        test -e "$eta_calib_path" && cp -v "$eta_calib_path" .
    fi
    for zs_calib_path in "$zs_calib_cemc" "$zs_calib_hcalin" "$zs_calib_hcalout"; do
        if [[ -n "$zs_calib_path" && "$zs_calib_path" != "none" && "$zs_calib_path" != "default" ]]; then
            if ! cp -v "$zs_calib_path" .; then
                echo "Error: Failed to copy ZS calib file $zs_calib_path for $file_calo at $(date) on $(hostname)" >&2
                echo "copy failure (zs_calib) for $zs_calib_path in $file_calo on $(hostname) at $(date)" >> "$submitDir/failures/failure-log.txt"
                exit 1
            fi
        fi
    done
    ls -lah
else
    echo "condor scratch NOT set" >&2
    exit 1
fi

# Check the local copies before running: the overlay production may still be writing an OutDir.
# A failed check is logged for resubmission.
check_out=$(root -b -l -q "$input_check_macro(\"$file_calofit\", \"$file_calo\", \"$file_jet\")" 2>&1)
check_exit=$?
echo "$check_out"
if [[ $check_exit -ne 0 ]] || ! grep -q "^INPUT_CHECK_OK" <<< "$check_out"; then
    reason=$(grep -m1 "^INPUT_CHECK_FAILED" <<< "$check_out" | sed 's/^INPUT_CHECK_FAILED: //')
    reason=${reason:-check macro exit code $check_exit}
    echo "Error: overlay inputs not ready for $input_global: $reason" >&2
    mkdir -p "$submitDir/failures"
    echo "input not ready for $input_global ($reason) on $(hostname) at $(date)" >> "$submitDir/failures/failure-log.txt"
    # 75 (EX_TEMPFAIL): the submit file has retry_until = 75, so condor does not retry this job right away;
    # resubmit it from the failure log once the production has finished the OutDir
    exit 75
fi

# print the environment - needed for debugging
printenv

mkdir -p "$outdir/hist" "$outdir/tree"

# mult is off in overlay mode
echo "Starting ROOT macro at $(date) on $(hostname)"
root -b -l -q "$f4a_macro(\"$file_calofit\", \"$file_zdc\", \"$file_sepd\", \"$calib_file\", \"$outdir/hist/$output\", \"$outdir/tree/$output_tree\", $do_flow, $nEvents, 0, 0, \"$dbtag\", \"$eta_calib_file\", \"\", $do_rcone, 0, \"\", $do_neg_energy_threshold, $neg_energy_threshold, \"${zs_calib_files[0]}\", \"${zs_calib_files[1]}\", \"${zs_calib_files[2]}\", $do_emfrac_calib, \"\", \"$file_calo\", \"$file_jet\", $jet_pt_min, $truth_jet_pt_min)"

root_exit=$?
if [ $root_exit -ne 0 ]; then
    echo "Error: ROOT macro failed with exit code $root_exit at $(date) on $(hostname)! Aborting transfer." >&2
    mkdir -p "$submitDir/failures"
    echo "ROOT failure (exit code $root_exit) for $file_calo on $(hostname) at $(date)" >> "$submitDir/failures/failure-log.txt"
    exit $root_exit
fi

echo "All Done and Transferring Files Back at $(date)"

# Define maximum retries and a counter
max_retries=5
count=0
success=0

while [ $count -lt $max_retries ]; do
    if cp -rv "$sample" "$submitDir"; then
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
    echo "CP transfer failure for $file_calo on $(hostname) at $(date)" >> "$submitDir/failures/failure-log.txt"
    exit 1
fi

echo "Finished successfully at $(date)"
