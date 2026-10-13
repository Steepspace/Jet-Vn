#!/usr/bin/env bash
set -euo pipefail

# ANSI color codes for readability
BOLD='\033[1m'
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RESET='\033[0m'

check_updates() {
  local repo_path="$1"
  local base_branch="$2"
  local target_branch="$3"
  local path_filter="$4"

  echo -e "\n${BOLD}${CYAN}=== Checking [${repo_path##*/}] : ${base_branch} -> ${target_branch} (${path_filter}) ===${RESET}"

  # Fetch all remotes quietly
  git -C "$repo_path" fetch --all --quiet

  # Check if there are unapplied commits on target_branch affecting the given path
  local commits
  commits=$(git -C "$repo_path" log --oneline --cherry-pick --right-only "${base_branch}...${target_branch}" -- "$path_filter")

  if [ -z "$commits" ]; then
    echo -e "${GREEN}Up to date. No new commits found.${RESET}"
  else
    echo -e "${YELLOW}Updates available:${RESET}"
    # Log commits with summary stats (files changed, insertions/deletions)
    git -C "$repo_path" log \
      --color=always \
      --stat \
      --cherry-pick \
      --right-only \
      "${base_branch}...${target_branch}" \
      -- "$path_filter"
  fi
}

# Has the upstream tree changed since we last integrated? Works after a squash merge,
# where --cherry-pick matching finds nothing even though the content landed.
check_pinned() {
  local repo_path="$1"
  local pinned="$2"
  local upstream="$3"
  local path_filter="$4"

  echo -e "\n${BOLD}${CYAN}=== Checking [${repo_path##*/}] : ${upstream} vs ${pinned} (${path_filter}) ===${RESET}"

  git -C "$repo_path" fetch --all --quiet

  if git -C "$repo_path" diff --quiet "$pinned" "$upstream" -- "$path_filter"; then
    echo -e "${GREEN}Up to date. No upstream changes.${RESET}"
  else
    echo -e "${YELLOW}Upstream changed:${RESET}"
    git -C "$repo_path" diff --stat --color=always "$pinned" "$upstream" -- "$path_filter"
  fi
}

# --- Repository Definitions ---
MINBIAS_DIR="$HOME/sPHENIX/coresoftware-MinBias"
JETBG_DIR="$HOME/sPHENIX/coresoftware-jetbackground-v2"
JETBG_PINNED="aabb69b48"   # upstream commit jetbackground-v2 is built on; bump after each integration

# 1. MinBias: trigger package against master
check_updates "$MINBIAS_DIR" "MinBias" "master" ":/offline/packages/trigger"

# 2. jetbackground-v2: upstream commits v2 doesn't have yet (PR #4465, tmengel's jstgtf03)
check_updates "$JETBG_DIR" "jetbackground-v2" "tanner/jstgtf03" ":/offline/packages/jetbackground"
check_updates "$JETBG_DIR" "jetbackground-v2" "tanner/jstgtf03" ":/offline/packages/jetbase"

# 3. jetbackground-v2: has upstream's jet content moved past the pinned commit?
check_pinned "$JETBG_DIR" "$JETBG_PINNED" "tanner/jstgtf03" ":/offline/packages/jetbackground"
check_pinned "$JETBG_DIR" "$JETBG_PINNED" "tanner/jstgtf03" ":/offline/packages/jetbase"

echo -e "\n${BOLD}Done.${RESET}"


