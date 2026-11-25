// Input check of one data+MC overlay job (genFun4All_BkgSub_Overlay.sh), run on the local copies before Fun4All_BkgSub.
//
// The overlay production can still be writing (or rewriting) an OutDir when the job copies it, and a copy taken
// mid-write is truncated. Fun4All does not always fail on such a file (it can finish with no events), so the job
// checks its inputs first:
//   - every file opens, is not a zombie and needed no recovery (ROOT recovers files that were not closed);
//   - DST_CALO and DST_TRUTH_JET hold the same number of events (> 0): they are read together in file order;
//   - DST_CALOFITTING holds at least as many (it may hold data events without an overlay partner);
//   - DST_TRUTH_JET carries PHGenIntegral (the luminosity weights).
//
// Prints "INPUT_CHECK_OK: ..." and exits 0, or prints "INPUT_CHECK_FAILED: <reason>" and exits 1.
//
// Usage: root -b -l -q 'check_overlay_inputs.C("DST_CALOFITTING-...root", "DST_CALO_...root", "DST_TRUTH_JET_...root")'

#include <TFile.h>
#include <TSystem.h>
#include <TTree.h>

#include <format>
#include <iostream>
#include <memory>
#include <string>

namespace
{
  // Entries of the event tree T, or -1 with the reason set
  long long event_entries(const std::string &name, bool need_gen_integral, std::string &reason)
  {
    std::unique_ptr<TFile> file(TFile::Open(name.c_str(), "READ"));
    if (!file || file->IsZombie())
    {
      reason = name + ": zombie";
      return -1;
    }
    if (file->TestBit(TFile::kRecovered))
    {
      reason = name + ": recovered (not closed, still being written?)";
      return -1;
    }

    auto *tree = file->Get<TTree>("T");
    if (!tree)
    {
      reason = name + ": no T tree";
      return -1;
    }

    if (need_gen_integral)
    {
      auto *run_tree = file->Get<TTree>("T1");
      if (!run_tree || !run_tree->GetBranch("RUN#PHGenIntegral"))
      {
        reason = name + ": no RUN#PHGenIntegral";
        return -1;
      }
    }

    return tree->GetEntries();
  }
}  // namespace

void check_overlay_inputs(const std::string &calofit, const std::string &calo, const std::string &truth_jet)
{
  std::string reason;

  const long long n_calofit = event_entries(calofit, false, reason);
  const long long n_calo = reason.empty() ? event_entries(calo, false, reason) : -1;
  const long long n_truth_jet = reason.empty() ? event_entries(truth_jet, true, reason) : -1;

  if (reason.empty())
  {
    if (n_calo == 0)
    {
      reason = calo + ": no events";
    }
    else if (n_calo != n_truth_jet)
    {
      reason = std::format("DST_CALO {} vs DST_TRUTH_JET {} events", n_calo, n_truth_jet);
    }
    else if (n_calofit < n_calo)
    {
      reason = std::format("DST_CALOFITTING {} < DST_CALO {} events", n_calofit, n_calo);
    }
  }

  if (!reason.empty())
  {
    std::cout << "INPUT_CHECK_FAILED: " << reason << std::endl;
    gSystem->Exit(1);
  }

  std::cout << std::format("INPUT_CHECK_OK: {} / {} / {} events (calofit / calo / truth jet)", n_calofit, n_calo, n_truth_jet) << std::endl;
  gSystem->Exit(0);
}
