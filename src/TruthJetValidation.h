// Tell emacs that this is a C++ source
//  -*- C++ -*-.
#pragma once

// -- sPHENIX
#include <fun4all/SubsysReco.h>

// -- c++
#include <string>
#include <vector>

class PHCompositeNode;
class JetContainer;
class TH1;

// Stores truth jets (data+MC overlay) in the TreeFiller tree.
// Truth <-> reco matching is left to the analysis.
// Also stores the generator bookkeeping (PHGenIntegral) in the histogram h_gen_integral (hadd-able)
// and, when the MC inputs live under their own top node, checks that every MC event is the one
// embedded into the data event under the data top node. Data events with no MC event embedded
// (dropped by the overlay production) are skipped: the MC inputs are pushed back by one event.
class TruthJetValidation : public SubsysReco
{
 public:
  explicit TruthJetValidation(const std::string &name = "TruthJetValidation");

  int Init(PHCompositeNode *topNode) override;
  int InitRun(PHCompositeNode *topNode) override;
  int process_event(PHCompositeNode *topNode) override;
  // Clean up internals after each event.
  int ResetEvent(PHCompositeNode *topNode) override;
  int End(PHCompositeNode *topNode) override;

  // h_gen_integral bins (all additive across jobs)
  enum GenBin
  {
    kLumiFile = 1,         // PHGenIntegral integrated luminosity of the input file [pb^-1]
    kNProcessedFile,       // PHGenIntegral events processed (written to the input file)
    kNGenAcceptedFile,     // PHGenIntegral events accepted by the generator (before the jet trigger)
    kSumWeightFile,        // PHGenIntegral sum of generator weights
    kNEventsJob,           // events analyzed by this job
    kLumiJob,              // integrated luminosity of the analyzed events [pb^-1]
    kNFiles                // number of input files
  };

  // data top node to check the MC events against (empty: no check)
  void set_data_topnode(const std::string &name) { m_data_topnode = name; }
  // no-sync MC input managers, pushed back to skip data events missing from the overlay (empty: abort instead)
  void set_mc_input_managers(const std::vector<std::string> &names) { m_mc_input_managers = names; }

  void set_do_r02(bool b = true) { m_do_r02 = b; }
  void set_do_r03(bool b = true) { m_do_r03 = b; }
  // leading R=0.4 truth jet pT (PHPy8JetTrigger variable, used to stitch pT-hat samples)
  void set_do_max_pt_r04(bool b = true) { m_do_max_pt_r04 = b; }
  void set_jet_pt_min(double pt_min) { m_jet_pt_min_cut = pt_min; }
  void set_jet_eta_max(double eta_max) { m_jet_eta_max_cut = eta_max; }

  void set_jet_radii(const std::vector<float>& radii);

  bool get_do_r02() const { return m_do_r02; }
  bool get_do_r03() const { return m_do_r03; }
  bool get_do_max_pt_r04() const { return m_do_max_pt_r04; }
  double get_jet_pt_min() const { return m_jet_pt_min_cut; }
  double get_jet_eta_max() const { return m_jet_eta_max_cut; }

 private:
  struct JetData
  {
    std::vector<double> pt;   // NOLINT(misc-non-private-member-variables-in-classes)
    std::vector<double> e;    // NOLINT(misc-non-private-member-variables-in-classes)
    std::vector<double> phi;  // NOLINT(misc-non-private-member-variables-in-classes)
    std::vector<double> eta;  // NOLINT(misc-non-private-member-variables-in-classes)
    double max_pt{0};         // NOLINT(misc-non-private-member-variables-in-classes)

    void clear()
    {
      pt.clear();
      e.clear();
      phi.clear();
      eta.clear();
      max_pt = 0;
    }
  };

  void fill_jets(JetContainer *jets, JetData &jd) const;

  JetData m_truth_r02;
  JetData m_truth_r03;
  double m_max_pt_truth_r04{0};

  TH1 *m_h_gen_integral{nullptr};
  double m_lumi_file{0};        // pb^-1
  double m_nprocessed_file{0};
  double m_nevents_job{0};
  unsigned int m_nskipped_data{0};  // data events with no MC event embedded

  std::string m_data_topnode;
  std::vector<std::string> m_mc_input_managers;

  std::string m_truthJetName_r02{"AntiKt_Truth_r02"};
  std::string m_truthJetName_r03{"AntiKt_Truth_r03"};
  std::string m_truthJetName_r04{"AntiKt_Truth_r04"};

  double m_jet_pt_min_cut{5};    // GeV
  double m_jet_eta_max_cut{1.5};

  bool m_do_r02{true};
  bool m_do_r03{true};
  bool m_do_max_pt_r04{true};
};
