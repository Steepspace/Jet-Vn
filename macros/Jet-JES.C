// -- utils
#include "myUtils.C"

// ====================================================================
// Standard C++ Includes
// ====================================================================
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numbers>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

// ====================================================================
// ROOT Includes
// ====================================================================
#include <TChain.h>
#include <TFile.h>
#include <TH1.h>
#include <TH2.h>
#include <TProfile.h>
#include <TROOT.h>

// ====================================================================
// The Analysis Class
// ====================================================================
// Jet energy scale in MC (data+MC overlay, p+p, PYTHIA8 embedded in HIJING): matches the truth jets to the
// reconstructed jets and fills the response pT_reco / pT_truth vs pT_truth in the eta bins of the legacy eta
// dependent JES. One pT-hat sample per job: the event is kept if the leading truth R=0.4 jet pT is in the sample's
// window, and the samples are weighted (1 / IntegratedLumi_job of h_gen_integral) and summed afterwards.
// Modes:
//  OVERLAY (0): data+MC overlay trees made with flow (do_flow != 0): iter / iternf jets; centrality, Calo-MBD,
//               flow failure and calo v2 selection.
//  PP      (1): p+p MC trees (Fun4All_PP_JES): unsub and iterative (no flow) jets; a single centrality class
//               (0-100) and no event selection besides the window (|z| < 10 cm is applied in Fun4All).
//  NOFLOW  (2): Au+Au trees made without flow (do_flow = 0): HIJING embedded MC (Fun4All_HIJING_JES) or data+MC
//               overlay; iterative (no flow) jets only; centrality selection, Calo-MBD selection only if a file
//               is given (not "none").
//  MB      (3): minimum-bias HIJING trees without truth jets (Fun4All_HIJING_JES with flist_truth_jet "none"): the
//               NOFLOW selection, but no truth branches, no window and no matching. Only the event hists, hCentrality,
//               h2RecoEta and the OHCal fraction templates (match state "incl") are filled.
// Trees made with do_flow = 0 store the no-flow iterative jets under "iter": the PP, NOFLOW and MB modes read those
// branches and name their histograms "iternf", as the OVERLAY mode does for the no-flow jets.
// Matching checks:
//  - trees with the true vertex (zvtx_truth, Fun4All_PP_JES with G4Hits): truth and matched jets vs
//    dz = z_MBD - z_truth, so the matching efficiency can be plotted vs the vertex error.
//  - dump_pt_min > 0: every fiducial truth jet with pT >= dump_pt_min that fails the matching is written, with the
//    event's truth and reco jets and the vertices, to <output_dir>/Jet-JES_<input>_failed_matches.txt.
// OHCal fraction templates (tasks/jet-ohcal-frac-templates.md), only for trees with ohcal_frac_<ue>_<r>:
//  h2OHCalFracPt_<r>_<ue>_<all|lead>_<matched|unmatched>_cent_<lo>_<hi>: x vs reco raw pT of the reco jets with E > 0,
//  |eta| < eta_max and x finite (all) and of the highest raw pT one (lead), split by whether the jet is matched to a truth
//  jet. That matching is separate from the JES one: one to one as well (truth jets by decreasing pT, closest free reco jet
//  within Delta R < 0.75 R), but over all stored truth jets, not only the fiducial ones. MB mode: match state "incl"
//  (no truth).
class JetJES
{
 public:
  enum class Mode : std::uint8_t
  {
    OVERLAY = 0,
    PP = 1,
    NOFLOW = 2,
    MB = 3
  };

  JetJES(std::string input_file, std::string calo_mbd_file, long long events, std::string output_dir, Mode mode = Mode::OVERLAY)
    : m_input_file(std::move(input_file))
    , m_calo_mbd_file(std::move(calo_mbd_file))
    , m_events_to_process(events)
    , m_output_dir(std::move(output_dir))
    , m_mode(mode)
    , m_is_pp(mode == Mode::PP)
    , m_has_truth(mode != Mode::MB)
    , m_ue_list(ue_list(mode))
    , m_frac_match(m_has_truth ? std::vector<std::string_view>{"matched", "unmatched"} : std::vector<std::string_view>{"incl"})
  {
  }

  static bool has_calo_mbd_file(const std::string &calo_mbd_file) { return !calo_mbd_file.empty() && calo_mbd_file != "none"; }

  static std::string_view mode_name(Mode mode)
  {
    switch (mode)
    {
    case Mode::PP:
      return "p+p MC";
    case Mode::NOFLOW:
      return "Au+Au without flow (HIJING embedded MC or no-flow data+MC overlay)";
    case Mode::MB:
      return "minimum-bias HIJING without truth (OHCal fraction templates only)";
    case Mode::OVERLAY:
    default:
      return "data+MC overlay";
    }
  }

  // Calo-MBD cut: OVERLAY always; NOFLOW and MB only with a file; PP never
  static bool uses_calo_mbd(Mode mode, const std::string &calo_mbd_file)
  {
    return mode == Mode::OVERLAY || ((mode == Mode::NOFLOW || mode == Mode::MB) && has_calo_mbd_file(calo_mbd_file));
  }

  void run()
  {
    if (uses_calo_mbd(m_mode, m_calo_mbd_file))
    {
      setup_calo_mbd_cut();
    }
    setup_chain();
    init_hists();
    open_dump();
    process_events();
    save_results();
  }

  // leading truth R=0.4 jet pT window [min, max), max < 0: no upper edge
  void set_lead_truth_pt_min(double pt) { m_lead_truth_pt_min = pt; }
  void set_lead_truth_pt_max(double pt) { m_lead_truth_pt_max = pt; }
  double get_lead_truth_pt_min() const { return m_lead_truth_pt_min; }
  double get_lead_truth_pt_max() const { return m_lead_truth_pt_max; }

  void set_verbosity(int verbosity) { m_verbosity = verbosity; }
  int get_verbosity() const { return m_verbosity; }

  void set_sigma_cut(double sigma) { m_sigma_cut = sigma; }
  double get_sigma_cut() const { return m_sigma_cut; }

  // failed-match dump threshold on the truth jet pT [GeV], <= 0: no dump
  void set_dump_pt_min(double pt) { m_dump_pt_min = pt; }
  double get_dump_pt_min() const { return m_dump_pt_min; }

 private:
  static constexpr double m_cent_high = 59.5;

  static constexpr std::array<double, 7> m_cent_bins = {0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0};
  static constexpr size_t m_num_cent_bins = m_cent_bins.size() - 1;

  // p+p: one class, every event
  static constexpr std::array<double, 2> m_cent_bins_pp = {0.0, 100.0};

  // JetCalib eta dependent JES: [-1.2+R, 1.2-R] split in 4 bins (|z| < 30 cm), cut at the jet fiducial |eta| < eta_max
  static constexpr size_t m_num_eta_bins = 4;

  struct RadiusConfig
  {
    std::string_view tag;
    double radius;
    double eta_max;
    std::array<double, m_num_eta_bins + 1> eta_edges;
  };

  static constexpr std::array<RadiusConfig, 2> m_radii = {{
      {"r02", 0.2, 0.9, {-0.9, -0.5, 0.0, 0.5, 0.9}},
      {"r03", 0.3, 0.8, {-0.8, -0.45, 0.0, 0.45, 0.8}}
  }};

  // UE: iter -> Iterative Method, iternf -> Iterative Method w/o flow modulation, unsub -> no subtraction.
  // name: in the histogram names; branch: in the tree branch names. Trees made with do_flow = 0 (p+p, HIJING,
  // no-flow overlay) store the no-flow iterative jets as "iter".
  struct UEConfig
  {
    std::string_view name;
    std::string_view branch;
  };

  static constexpr size_t m_max_ue = 2;

  static std::vector<UEConfig> ue_list(Mode mode)
  {
    switch (mode)
    {
    case Mode::PP:
      return {{"unsub", "unsub"}, {"iternf", "iter"}};
    case Mode::NOFLOW:
    case Mode::MB:
      return {{"iternf", "iter"}};
    case Mode::OVERLAY:
    default:
      return {{"iter", "iter"}, {"iternf", "iternf"}};
    }
  }

  // Matching radius: Delta R < m_match_radius_frac * R
  static constexpr double m_match_radius_frac = 0.75;

  // Jet-Anav3 v3 selection: |calo v2| < m_calo_v2_max (iter, also for iternf) and reco jet energy > 0
  static constexpr float m_calo_v2_max = 0.48F;

  template <typename T>
  using EtaCentArray = std::array<std::array<T *, m_num_cent_bins>, m_num_eta_bins>;

  // OHCal fraction templates: jet sets (every jet, leading jet) x truth match states (m_frac_match: matched /
  // unmatched, or incl without truth), as in the data C4 (Jet-Dijet.C)
  static constexpr std::array<std::string_view, 2> m_frac_sets = {"all", "lead"};
  static constexpr size_t m_max_frac_match = 2;
  static constexpr int m_bins_frac = 60;
  static constexpr double m_frac_low = -0.25;
  static constexpr double m_frac_high = 1.25;
  // reco raw pT in 1 GeV bins, re-binned offline to the C4 classes (7, 10, 15, 20, 30, 40, 60, 150, 300)
  static constexpr int m_bins_frac_pt = 300;
  static constexpr double m_frac_pt_low = 0;
  static constexpr double m_frac_pt_high = 300;

  // [set][match][cent]
  using FracArray = std::array<std::array<std::array<TH2 *, m_num_cent_bins>, m_max_frac_match>, m_frac_sets.size()>;

  struct ResponseHistSet
  {
    EtaCentArray<TH2> h2Resp_raw{};
    EtaCentArray<TH2> h2Resp_calib{};
    EtaCentArray<TH1> hTruthMatched{};
    TH2 *h2DeltaEta{nullptr};
    TH2 *h2DeltaR{nullptr};
    // [eta], matched truth jets: pT_truth vs z_MBD - z_truth (only with the truth vertex)
    std::array<TH2 *, m_num_eta_bins> h2TruthMatchedDz{};
    // all reco jets, as h2JetEta in Jet-Anav3
    TH2 *h2RecoEta_raw{nullptr};
    TH2 *h2RecoEta_calib{nullptr};
    // OHCal fraction x vs reco raw pT (only for trees with ohcal_frac)
    FracArray h2OHCalFracPt{};
  };

  struct AnalysisHists
  {
    TH1 *hEvent{nullptr};
    TH1 *hCentrality{nullptr};

    TH1 *hMaxPtTruthR04_all{nullptr};
    TH1 *hMaxPtTruthR04{nullptr};
    TH1 *hMaxPtTruthR04_window{nullptr};

    // [radius]
    std::array<EtaCentArray<TH1>, m_radii.size()> hTruth{};
    // [radius], all truth jets
    std::array<TH2 *, m_radii.size()> h2TruthEta{};
    // [radius][eta], fiducial truth jets: pT_truth vs z_MBD - z_truth (only with the truth vertex)
    std::array<std::array<TH2 *, m_num_eta_bins>, m_radii.size()> h2TruthDz{};
    // events: z_truth vs z_MBD, and z_MBD - z_truth vs the leading truth R=0.4 jet pT (only with the truth vertex)
    TH2 *h2ZvtxTruth{nullptr};
    TH2 *h2DzLeadPt{nullptr};
    // [ue][radius]
    std::array<std::array<ResponseHistSet, m_radii.size()>, m_max_ue> resp{};
  };

  AnalysisHists m_hists;

  struct JetData
  {
    std::vector<double> *pt{nullptr};        // NOLINT(misc-non-private-member-variables-in-classes)
    std::vector<double> *pt_calib{nullptr};  // NOLINT(misc-non-private-member-variables-in-classes)
    std::vector<double> *phi{nullptr};       // NOLINT(misc-non-private-member-variables-in-classes)
    std::vector<double> *eta{nullptr};       // NOLINT(misc-non-private-member-variables-in-classes)
    std::vector<double> *e{nullptr};         // NOLINT(misc-non-private-member-variables-in-classes)
    std::vector<double> *ohcal_frac{nullptr};  // NOLINT(misc-non-private-member-variables-in-classes), reco only, optional
  };

  struct EventData
  {
    friend class JetJES;

   private:
    int event{0};
    double zvtx{0.0};
    double zvtx_truth{0.0};  // only with the truth vertex
    double centrality{0.0};
    double emcal_energy{0};
    double ihcal_energy{0};
    double ohcal_energy{0};

    double mbd_charge_south{0.0};
    double mbd_charge_north{0.0};

    bool is_flow_failure_iter{false};
    float calo_v2_iter{0};

    // leading truth R=0.4 jet pT, the pT-hat sample generator trigger variable
    double max_pt_truth_r04{0};

    // [radius], truth jets have no pt_calib and e is not read
    std::array<JetData, m_radii.size()> truth;
    // [ue][radius]
    std::array<std::array<JetData, m_radii.size()>, m_max_ue> reco;
  };

  // fiducial truth jet: index in the event and eta bin
  struct TruthJet
  {
    size_t idx{0};
    size_t ieta{0};
  };

  // --- Member Variables ---
  EventData m_event_data;
  std::unique_ptr<TChain> m_chain;

  // Configuration stored as members
  std::string m_input_file;
  std::string m_calo_mbd_file;
  long long m_events_to_process;
  std::string m_output_dir;
  Mode m_mode{Mode::OVERLAY};
  bool m_is_pp{false};
  // truth jets in the trees (every mode but MB)
  bool m_has_truth{true};
  std::vector<UEConfig> m_ue_list;
  // OHCal fraction template match states of this mode
  std::vector<std::string_view> m_frac_match;
  int m_verbosity{0};

  double m_lead_truth_pt_min{0}; /*GeV*/
  double m_lead_truth_pt_max{-1}; /*GeV*/

  std::map<std::string, int> m_ctr;

  enum class EventType : std::uint8_t
  {
    ALL,
    CENT,
    CALO_MBD,
    FLOW_PASS_ITER,
    CALO_V2,
    TRUTH_WINDOW
  };

  // event selection steps of this mode, in hEvent bin order
  std::vector<EventType> m_event_steps;

  // Hists
  std::map<std::string, std::unique_ptr<TH1>> m_hists1D;
  std::map<std::string, std::unique_ptr<TH2>> m_hists2D;

  double m_sigma_cut{3.5};
  bool m_has_calo_mbd_cut{false};

  // OHCal fraction templates: every reco branch ohcal_frac_<ue>_<r> is in the trees
  bool m_has_ohcal_frac{false};

  // matching checks
  bool m_has_truth_vertex{false};
  double m_dump_pt_min{-1}; /*GeV*/
  std::ofstream m_dump;
  long long m_entry{0};
  std::unique_ptr<TProfile> m_profile_calo_mbd{nullptr};

  // per event buffers
  std::vector<TruthJet> m_truth_jets;
  std::vector<bool> m_reco_used;
  // [truth jet index]: matched reco jet (-1: none), closest free reco jet when it was tried and its Delta R
  std::vector<int> m_truth_match;
  std::vector<int> m_truth_closest;
  std::vector<double> m_truth_closest_dr;
  // OHCal fraction templates: all truth jets by decreasing pT, reco jets taken, and per reco jet whether it is matched
  std::vector<size_t> m_frac_truth_order;
  std::vector<bool> m_frac_reco_used;
  std::vector<bool> m_frac_reco_matched;

  // --- Private Helper Methods ---
  void setup_chain();
  void setup_calo_mbd_cut();
  void init_hists();

  bool check_CaloMBD() const;
  bool check_truth_window() const;
  void fill_event(EventType type) const;
  size_t num_cent_bins() const { return m_is_pp ? 1 : m_num_cent_bins; }
  std::pair<int, int> cent_range(size_t icent) const;
  std::optional<size_t> find_cent_bin(double cent) const;
  static std::optional<size_t> find_eta_bin(double eta, const RadiusConfig &rc);
  static double delta_phi(double phi1, double phi2);
  static double delta_r(double eta1, double phi1, double eta2, double phi2);
  // closest reco jet not yet used to (eta, phi) and its Delta R; nullopt if every reco jet is used
  static std::optional<std::pair<size_t, double>> closest_free_reco(double eta, double phi, const JetData &reco, const std::vector<bool> &used);

  void process_jets(size_t icent);
  void fill_ohcal_frac(size_t ir, size_t iue, size_t icent);
  void process_events();
  void print_event_info(long long event_idx) const;
  void open_dump();
  void dump_failed_matches(size_t ir, size_t iue);

  void save_results() const;
};

// ====================================================================
// Implementation of the Class Methods
// ====================================================================
void JetJES::setup_chain()
{
  std::cout << "Processing... setup_chain" << std::endl;

  m_chain = myUtils::setupTChain(m_input_file, "T");

  if (m_chain == nullptr)
  {
    throw std::runtime_error(std::format("Error in TChain Setup from file: {}", m_input_file));
  }

  m_chain->SetBranchStatus("*", false);

  auto set_branch = [this](const std::string &name, auto *address)
  {
    if (!m_chain->GetBranch(name.c_str()))
    {
      throw std::runtime_error(std::format("Could not find Branch '{}' in file '{}'", name, m_input_file));
    }
    m_chain->SetBranchStatus(name.c_str(), true);
    m_chain->SetBranchAddress(name.c_str(), address);
  };

  auto &ed = m_event_data;
  set_branch("event", &ed.event);
  set_branch("zvtx", &ed.zvtx);
  if (m_has_truth)
  {
    set_branch("max_pt_truth_r04", &ed.max_pt_truth_r04);
  }
  // true vertex: only in trees made with the G4Hits (Fun4All_PP_JES)
  m_has_truth_vertex = m_has_truth && m_chain->GetBranch("zvtx_truth") != nullptr;
  if (m_has_truth_vertex)
  {
    set_branch("zvtx_truth", &ed.zvtx_truth);
  }
  std::cout << std::format("Truth vertex (zvtx_truth): {}\n", m_has_truth_vertex ? "found, filling the dz histograms" : "not in the trees");
  // p+p: no centrality, Calo-MBD or flow selection (the p+p trees have no calo energy / MBD charge branches)
  if (!m_is_pp)
  {
    set_branch("centrality", &ed.centrality);
  }
  if (m_has_calo_mbd_cut)
  {
    set_branch("emcal_energy", &ed.emcal_energy);
    set_branch("ihcal_energy", &ed.ihcal_energy);
    set_branch("ohcal_energy", &ed.ohcal_energy);
    set_branch("mbd_charge_south", &ed.mbd_charge_south);
    set_branch("mbd_charge_north", &ed.mbd_charge_north);
  }
  // flow selection only for trees made with flow modulation
  if (m_mode == Mode::OVERLAY)
  {
    set_branch("is_flow_failure_iter", &ed.is_flow_failure_iter);
    set_branch("calo_v2_iter", &ed.calo_v2_iter);
  }

  // OHCal fraction templates: only if every reco jet collection has its ohcal_frac branch (trees made with do_timing)
  m_has_ohcal_frac = true;
  for (const auto &rc : m_radii)
  {
    for (const auto &ue : m_ue_list)
    {
      m_has_ohcal_frac = m_has_ohcal_frac && m_chain->GetBranch(std::format("ohcal_frac_{}_{}", ue.branch, rc.tag).c_str()) != nullptr;
    }
  }
  std::cout << std::format("OHCal fraction (ohcal_frac): {}\n", m_has_ohcal_frac ? "found, filling the x templates" : "not in the trees, no x templates");
  // without truth the x templates are the only output
  if (!m_has_truth && !m_has_ohcal_frac)
  {
    throw std::runtime_error(std::format("MB mode needs the ohcal_frac branches, not in '{}'", m_input_file));
  }

  for (size_t ir = 0; ir < m_radii.size(); ++ir)
  {
    std::string_view r = m_radii[ir].tag;

    if (m_has_truth)
    {
      JetData &truth = ed.truth[ir];
      set_branch(std::format("pt_truth_{}", r), &truth.pt);
      set_branch(std::format("phi_truth_{}", r), &truth.phi);
      set_branch(std::format("eta_truth_{}", r), &truth.eta);
    }

    for (size_t iue = 0; iue < m_ue_list.size(); ++iue)
    {
      std::string_view ue = m_ue_list[iue].branch;
      JetData &reco = ed.reco[iue][ir];
      set_branch(std::format("pt_{}_{}", ue, r), &reco.pt);
      set_branch(std::format("pt_calib_{}_{}", ue, r), &reco.pt_calib);
      set_branch(std::format("phi_{}_{}", ue, r), &reco.phi);
      set_branch(std::format("eta_{}_{}", ue, r), &reco.eta);
      set_branch(std::format("e_{}_{}", ue, r), &reco.e);
      if (m_has_ohcal_frac)
      {
        set_branch(std::format("ohcal_frac_{}_{}", ue, r), &reco.ohcal_frac);
      }
    }
  }

  std::cout << "Finished... setup_chain" << std::endl;
}

void JetJES::setup_calo_mbd_cut()
{
  if (m_calo_mbd_file.empty())
  {
    throw std::runtime_error("Calo-MBD input file is required but was not provided.");
  }

  if (!std::filesystem::exists(m_calo_mbd_file))
  {
    throw std::runtime_error(std::format("Calo-MBD file does not exist: {}", m_calo_mbd_file));
  }

  auto tfile = std::unique_ptr<TFile>(TFile::Open(m_calo_mbd_file.c_str(), "READ"));
  if (!tfile || tfile->IsZombie())
  {
    throw std::runtime_error(std::format("Could not open Calo-MBD file: {}", m_calo_mbd_file));
  }

  TH2 *h2 = dynamic_cast<TH2 *>(tfile->Get("h2TotalCaloE_MBD"));
  if (!h2)
  {
    throw std::runtime_error(std::format("h2TotalCaloE_MBD not found in {}", m_calo_mbd_file));
  }

  // Profile X with "s" option so error is spread (standard deviation)
  TProfile *pfx = h2->ProfileX("pfx_calo_mbd_cut", 1, -1, "s");
  if (!pfx)
  {
    throw std::runtime_error(std::format("Failed to create ProfileX from h2TotalCaloE_MBD in {}", m_calo_mbd_file));
  }

  pfx->SetDirectory(nullptr);
  m_profile_calo_mbd = std::unique_ptr<TProfile>(pfx);
  m_has_calo_mbd_cut = true;

  std::cout << std::format("Successfully loaded h2TotalCaloE_MBD from {} and created ProfileX with 's' option (sigma cut: {:.2f})\n", m_calo_mbd_file, m_sigma_cut);
}

void JetJES::init_hists()
{
  int bins_pt = 50;
  double pt_low = 0;
  double pt_high = 100;

  int bins_resp = 150;
  double resp_low = 0;
  double resp_high = 3;

  int bins_lead_pt = 150;
  double lead_pt_low = 0;
  double lead_pt_high = 150;

  int bins_deta = 60;
  double deta_low = -0.3;
  double deta_high = 0.3;

  int bins_dr = 60;
  double dr_low = 0;
  double dr_high = 0.6;

  // same eta bin width as h2JetEta in Jet-Anav3, wide enough for the truth jets (|eta| < 1.5)
  int bins_pt_eta2D = 80;
  double pt_eta2D_low = 0;
  double pt_eta2D_high = 80;

  int bins_eta = 120;
  double eta_low = -1.5;
  double eta_high = 1.5;

  // z_MBD - z_truth: the true vertex is broad (~66 cm RMS), so a wrong MBD vertex can be far off
  int bins_dz = 300;
  double dz_low = -150;
  double dz_high = 150;

  int bins_zmbd = 40;
  double zmbd_low = -10;
  double zmbd_high = 10;

  int bins_ztruth = 300;
  double ztruth_low = -150;
  double ztruth_high = 150;

  const std::map<EventType, std::string> event_labels = {
      {EventType::ALL, "All"},
      {EventType::CENT, "Centrality 0-60%"},
      {EventType::CALO_MBD, "Good Calo-MBD"},
      {EventType::FLOW_PASS_ITER, "Flow Pass Iter"},
      {EventType::CALO_V2, std::format("|Calo v_{{2}}| < {}", m_calo_v2_max)},
      {EventType::TRUTH_WINDOW, std::format("Lead Truth R=0.4 p_{{T}} Window [{}, {})", m_lead_truth_pt_min,
                                            m_lead_truth_pt_max < 0 ? "#infty" : std::format("{}", m_lead_truth_pt_max))}};

  switch (m_mode)
  {
  case Mode::PP:
    m_event_steps = {EventType::ALL, EventType::TRUTH_WINDOW};
    break;
  case Mode::NOFLOW:
  case Mode::MB:
    m_event_steps = {EventType::ALL, EventType::CENT};
    if (m_has_calo_mbd_cut)
    {
      m_event_steps.push_back(EventType::CALO_MBD);
    }
    // MB: no truth, no window
    if (m_has_truth)
    {
      m_event_steps.push_back(EventType::TRUTH_WINDOW);
    }
    break;
  case Mode::OVERLAY:
  default:
    m_event_steps = {EventType::ALL, EventType::CENT, EventType::CALO_MBD, EventType::FLOW_PASS_ITER, EventType::CALO_V2, EventType::TRUTH_WINDOW};
    break;
  }

  int bins_event = static_cast<int>(m_event_steps.size());
  m_hists1D["hEvent"] = std::make_unique<TH1F>("hEvent", "Event Type; Type; Events", bins_event, 0, bins_event);
  for (size_t i = 0; i < m_event_steps.size(); ++i)
  {
    m_hists1D["hEvent"]->GetXaxis()->SetBinLabel(static_cast<int>(i + 1), event_labels.at(m_event_steps[i]).c_str());
  }

  if (!m_is_pp)
  {
    m_hists1D["hCentrality"] = std::make_unique<TH1F>("hCentrality", "Selected Events; Centrality [%]; Events", 60, -0.5, 59.5);
  }

  m_hists.hEvent = m_hists1D["hEvent"].get();
  m_hists.hCentrality = m_is_pp ? nullptr : m_hists1D["hCentrality"].get();

  if (m_has_truth)
  {
    m_hists1D["hMaxPtTruthR04_all"] = std::make_unique<TH1F>("hMaxPtTruthR04_all", "All Events; Leading Truth R=0.4 Jet p_{T} [GeV]; Events", bins_lead_pt, lead_pt_low, lead_pt_high);
    m_hists1D["hMaxPtTruthR04"] = std::make_unique<TH1F>("hMaxPtTruthR04", "Event Selection (before window); Leading Truth R=0.4 Jet p_{T} [GeV]; Events", bins_lead_pt, lead_pt_low, lead_pt_high);
    m_hists1D["hMaxPtTruthR04_window"] = std::make_unique<TH1F>("hMaxPtTruthR04_window", "Event Selection + Window; Leading Truth R=0.4 Jet p_{T} [GeV]; Events", bins_lead_pt, lead_pt_low, lead_pt_high);
    m_hists.hMaxPtTruthR04_all = m_hists1D["hMaxPtTruthR04_all"].get();
    m_hists.hMaxPtTruthR04 = m_hists1D["hMaxPtTruthR04"].get();
    m_hists.hMaxPtTruthR04_window = m_hists1D["hMaxPtTruthR04_window"].get();
  }

  if (m_has_truth_vertex)
  {
    m_hists2D["h2ZvtxTruth"] = std::make_unique<TH2F>("h2ZvtxTruth", "Event Selection + Window; z_{MBD} [cm]; z_{truth} [cm]", bins_zmbd, zmbd_low, zmbd_high, bins_ztruth, ztruth_low, ztruth_high);
    m_hists2D["h2DzLeadPt"] = std::make_unique<TH2F>("h2DzLeadPt", "Event Selection + Window; Leading Truth R=0.4 Jet p_{T} [GeV]; z_{MBD} - z_{truth} [cm]", bins_lead_pt, lead_pt_low, lead_pt_high, bins_dz, dz_low, dz_high);
    m_hists.h2ZvtxTruth = m_hists2D["h2ZvtxTruth"].get();
    m_hists.h2DzLeadPt = m_hists2D["h2DzLeadPt"].get();
  }

  for (size_t ir = 0; ir < m_radii.size(); ++ir)
  {
    const RadiusConfig &rc = m_radii[ir];
    std::string_view r = rc.tag;

    if (m_has_truth)
    {
      std::string truth_eta_name = std::format("h2TruthEta_{}", r);
      m_hists2D[truth_eta_name] = std::make_unique<TH2F>(truth_eta_name.c_str(), std::format("R = {:.1f}, All Truth Jets; p_{{T}}^{{truth}} [GeV]; #eta^{{truth}}", rc.radius).c_str(), bins_pt_eta2D, pt_eta2D_low, pt_eta2D_high, bins_eta, eta_low, eta_high);
      m_hists.h2TruthEta[ir] = m_hists2D[truth_eta_name].get();
    }

    // vs the vertex error, all centralities (one TH2 per eta bin)
    if (m_has_truth_vertex)
    {
      for (size_t ieta = 0; ieta < m_num_eta_bins; ++ieta)
      {
        std::string label = std::format("R = {:.1f}, {:.2f} #leq #eta^{{truth}} < {:.2f}", rc.radius, rc.eta_edges[ieta], rc.eta_edges[ieta + 1]);
        std::string truth_dz_name = std::format("h2TruthDz_{}_eta{}", r, ieta);
        m_hists2D[truth_dz_name] = std::make_unique<TH2F>(truth_dz_name.c_str(), std::format("{}; p_{{T}}^{{truth}} [GeV]; z_{{MBD}} - z_{{truth}} [cm]", label).c_str(), bins_pt, pt_low, pt_high, bins_dz, dz_low, dz_high);
        m_hists.h2TruthDz[ir][ieta] = m_hists2D[truth_dz_name].get();

        for (size_t iue = 0; iue < m_ue_list.size(); ++iue)
        {
          std::string_view ue = m_ue_list[iue].name;
          std::string matched_dz_name = std::format("h2TruthMatchedDz_{}_{}_eta{}", r, ue, ieta);
          m_hists2D[matched_dz_name] = std::make_unique<TH2F>(matched_dz_name.c_str(), std::format("{}, {}, Matched; p_{{T}}^{{truth}} [GeV]; z_{{MBD}} - z_{{truth}} [cm]", ue, label).c_str(), bins_pt, pt_low, pt_high, bins_dz, dz_low, dz_high);
          m_hists.resp[iue][ir].h2TruthMatchedDz[ieta] = m_hists2D[matched_dz_name].get();
        }
      }
    }

    // truth and response hists (not in MB mode)
    for (size_t ieta = 0; m_has_truth && ieta < m_num_eta_bins; ++ieta)
    {
      for (size_t icent = 0; icent < num_cent_bins(); ++icent)
      {
        auto [cent_low, cent_high] = cent_range(icent);
        std::string suffix = std::format("eta{}_cent_{}_{}", ieta, cent_low, cent_high);
        std::string cent_label = m_is_pp ? std::string("p+p") : std::format("{} - {}%", cent_low, cent_high);
        std::string label = std::format("R = {:.1f}, {:.2f} #leq #eta^{{truth}} < {:.2f}, {}", rc.radius, rc.eta_edges[ieta], rc.eta_edges[ieta + 1], cent_label);

        std::string truth_name = std::format("hTruth_{}_{}", r, suffix);
        m_hists1D[truth_name] = std::make_unique<TH1F>(truth_name.c_str(), std::format("{}; p_{{T}}^{{truth}} [GeV]; Jets", label).c_str(), bins_pt, pt_low, pt_high);
        m_hists.hTruth[ir][ieta][icent] = m_hists1D[truth_name].get();

        for (size_t iue = 0; iue < m_ue_list.size(); ++iue)
        {
          std::string_view ue = m_ue_list[iue].name;
          ResponseHistSet &h = m_hists.resp[iue][ir];
          std::string ue_label = std::format("{}, {}", ue, label);

          std::string matched_name = std::format("hTruthMatched_{}_{}_{}", r, ue, suffix);
          m_hists1D[matched_name] = std::make_unique<TH1F>(matched_name.c_str(), std::format("{}; p_{{T}}^{{truth}} [GeV]; Matched Jets", ue_label).c_str(), bins_pt, pt_low, pt_high);
          h.hTruthMatched[ieta][icent] = m_hists1D[matched_name].get();

          for (const char *calib : {"raw", "calib"})
          {
            std::string resp_name = std::format("h2Resp_{}_{}_{}_{}", calib, r, ue, suffix);
            std::string resp_title = std::format("{}; p_{{T}}^{{truth}} [GeV]; p_{{T}}^{{reco, {}}} / p_{{T}}^{{truth}}", ue_label, calib);
            m_hists2D[resp_name] = std::make_unique<TH2F>(resp_name.c_str(), resp_title.c_str(), bins_pt, pt_low, pt_high, bins_resp, resp_low, resp_high);
            (std::string_view(calib) == "raw" ? h.h2Resp_raw : h.h2Resp_calib)[ieta][icent] = m_hists2D[resp_name].get();
          }
        }
      }
    }

    for (size_t iue = 0; iue < m_ue_list.size(); ++iue)
    {
      std::string_view ue = m_ue_list[iue].name;
      ResponseHistSet &h = m_hists.resp[iue][ir];

      if (m_has_truth)
      {
        std::string deta_name = std::format("h2DeltaEta_{}_{}", r, ue);
        m_hists2D[deta_name] = std::make_unique<TH2F>(deta_name.c_str(), std::format("R = {:.1f}, {}, Matched; p_{{T}}^{{truth}} [GeV]; #eta^{{reco}} - #eta^{{truth}}", rc.radius, ue).c_str(), bins_pt, pt_low, pt_high, bins_deta, deta_low, deta_high);
        h.h2DeltaEta = m_hists2D[deta_name].get();

        std::string dr_name = std::format("h2DeltaR_{}_{}", r, ue);
        m_hists2D[dr_name] = std::make_unique<TH2F>(dr_name.c_str(), std::format("R = {:.1f}, {}, Closest Reco Jet (before #DeltaR cut); p_{{T}}^{{truth}} [GeV]; #DeltaR", rc.radius, ue).c_str(), bins_pt, pt_low, pt_high, bins_dr, dr_low, dr_high);
        h.h2DeltaR = m_hists2D[dr_name].get();
      }

      for (const char *calib : {"raw", "calib"})
      {
        std::string reco_eta_name = std::format("h2RecoEta_{}_{}_{}", calib, r, ue);
        m_hists2D[reco_eta_name] = std::make_unique<TH2F>(reco_eta_name.c_str(), std::format("R = {:.1f}, {}, All Reco Jets; p_{{T}}^{{reco, {}}} [GeV]; #eta^{{reco}}", rc.radius, ue, calib).c_str(), bins_pt_eta2D, pt_eta2D_low, pt_eta2D_high, bins_eta, eta_low, eta_high);
        (std::string_view(calib) == "raw" ? h.h2RecoEta_raw : h.h2RecoEta_calib) = m_hists2D[reco_eta_name].get();
      }

      if (m_has_ohcal_frac)
      {
        for (size_t is = 0; is < m_frac_sets.size(); ++is)
        {
          for (size_t im = 0; im < m_frac_match.size(); ++im)
          {
            for (size_t icent = 0; icent < num_cent_bins(); ++icent)
            {
              auto [cent_low, cent_high] = cent_range(icent);
              std::string cent_label = m_is_pp ? std::string("p+p") : std::format("{} - {}%", cent_low, cent_high);
              std::string name = std::format("h2OHCalFracPt_{}_{}_{}_{}_cent_{}_{}", r, ue, m_frac_sets[is], m_frac_match[im], cent_low, cent_high);
              std::string title = std::format("R = {:.1f}, {}, |#eta| < {}, {} jets, {}, {}; OHCal Fraction; p_{{T}}^{{reco, raw}} [GeV]",
                                              rc.radius, ue, rc.eta_max, m_frac_sets[is], m_frac_match[im], cent_label);
              m_hists2D[name] = std::make_unique<TH2F>(name.c_str(), title.c_str(), m_bins_frac, m_frac_low, m_frac_high, m_bins_frac_pt, m_frac_pt_low, m_frac_pt_high);
              h.h2OHCalFracPt[is][im][icent] = m_hists2D[name].get();
            }
          }
        }
      }
    }
  }

  // Enable Sumw2
  auto enable = [](auto &...maps)
  {
    auto loop = [](auto &map)
    {
      for (auto &[key, hist] : map)
      {
        if (hist)
        {
          hist->Sumw2();
        }
      }
    };
    (loop(maps), ...);
  };

  enable(m_hists1D, m_hists2D);
}

bool JetJES::check_CaloMBD() const
{
  double total_energy = m_event_data.emcal_energy + m_event_data.ihcal_energy + m_event_data.ohcal_energy;
  double mbd_total = m_event_data.mbd_charge_south + m_event_data.mbd_charge_north;

  if (m_has_calo_mbd_cut && m_profile_calo_mbd)
  {
    int bin = m_profile_calo_mbd->FindBin(mbd_total);
    if (bin >= 1 && bin <= m_profile_calo_mbd->GetNbinsX())
    {
      double mean = m_profile_calo_mbd->GetBinContent(bin);
      double sigma = m_profile_calo_mbd->GetBinError(bin);
      return sigma > 0.0 && std::abs(total_energy - mean) <= m_sigma_cut * sigma;
    }
  }

  return false;
}

bool JetJES::check_truth_window() const
{
  double pt = m_event_data.max_pt_truth_r04;
  return pt >= m_lead_truth_pt_min && (m_lead_truth_pt_max < 0 || pt < m_lead_truth_pt_max);
}

void JetJES::fill_event(EventType type) const
{
  auto it = std::ranges::find(m_event_steps, type);
  if (it != m_event_steps.end())
  {
    m_hists.hEvent->Fill(static_cast<double>(std::distance(m_event_steps.begin(), it)));
  }
}

std::pair<int, int> JetJES::cent_range(size_t icent) const
{
  if (m_is_pp)
  {
    return {static_cast<int>(m_cent_bins_pp[0]), static_cast<int>(m_cent_bins_pp[1])};
  }
  return {static_cast<int>(m_cent_bins[icent]), static_cast<int>(m_cent_bins[icent + 1])};
}

std::optional<size_t> JetJES::find_cent_bin(double cent) const
{
  // p+p: every event is in the single class
  if (m_is_pp)
  {
    return 0;
  }

  if (cent > m_cent_high)
  {
    return std::nullopt;
  }

  for (size_t icent = 0; icent < m_num_cent_bins; ++icent)
  {
    if (cent >= m_cent_bins[icent] && (cent < m_cent_bins[icent + 1] || (icent == m_num_cent_bins - 1 && cent <= m_cent_bins[icent + 1])))
    {
      return icent;
    }
  }

  return std::nullopt;
}

std::optional<size_t> JetJES::find_eta_bin(double eta, const RadiusConfig &rc)
{
  if (std::abs(eta) >= rc.eta_max)
  {
    return std::nullopt;
  }

  // inner edges only: the outer bins extend to the fiducial cut
  auto it = std::upper_bound(rc.eta_edges.begin() + 1, rc.eta_edges.end() - 1, eta);
  return static_cast<size_t>(std::distance(rc.eta_edges.begin() + 1, it));
}

// phi1 - phi2 in [-pi, pi)
double JetJES::delta_phi(double phi1, double phi2)
{
  double dphi = std::fmod(phi1 - phi2 + std::numbers::pi, 2 * std::numbers::pi);
  if (dphi < 0)
  {
    dphi += 2 * std::numbers::pi;
  }
  return dphi - std::numbers::pi;
}

double JetJES::delta_r(double eta1, double phi1, double eta2, double phi2)
{
  return std::hypot(eta1 - eta2, delta_phi(phi1, phi2));
}

void JetJES::process_jets(size_t icent)
{
  const auto &ed = m_event_data;

  for (size_t ir = 0; ir < m_radii.size(); ++ir)
  {
    const RadiusConfig &rc = m_radii[ir];
    const JetData &truth = ed.truth[ir];

    // MB: no truth, only the reco jets (as h2JetEta in Jet-Anav3) and the x templates
    if (!m_has_truth)
    {
      for (size_t iue = 0; iue < m_ue_list.size(); ++iue)
      {
        const JetData &reco = ed.reco[iue][ir];
        const ResponseHistSet &h = m_hists.resp[iue][ir];
        if (!reco.pt || !reco.pt_calib || !reco.eta || !reco.e)
        {
          continue;
        }
        for (size_t jdx = 0; jdx < reco.pt->size(); ++jdx)
        {
          if (reco.e->at(jdx) > 0)
          {
            h.h2RecoEta_raw->Fill(reco.pt->at(jdx), reco.eta->at(jdx));
            h.h2RecoEta_calib->Fill(reco.pt_calib->at(jdx), reco.eta->at(jdx));
          }
        }
        fill_ohcal_frac(ir, iue, icent);
      }
      continue;
    }

    if (!truth.pt || !truth.phi || !truth.eta)
    {
      continue;
    }

    // fiducial truth jets, leading first so they get the first pick of the reco jets
    m_truth_jets.clear();
    for (size_t idx = 0; idx < truth.pt->size(); ++idx)
    {
      m_hists.h2TruthEta[ir]->Fill(truth.pt->at(idx), truth.eta->at(idx));

      auto ieta = find_eta_bin(truth.eta->at(idx), rc);
      if (ieta)
      {
        m_truth_jets.push_back({idx, *ieta});
        m_hists.hTruth[ir][*ieta][icent]->Fill(truth.pt->at(idx));
        if (m_has_truth_vertex)
        {
          m_hists.h2TruthDz[ir][*ieta]->Fill(truth.pt->at(idx), ed.zvtx - ed.zvtx_truth);
        }
      }
    }
    std::ranges::sort(m_truth_jets, [&](const TruthJet &a, const TruthJet &b)
                      { return truth.pt->at(a.idx) > truth.pt->at(b.idx); });

    double dr_max = m_match_radius_frac * rc.radius;

    for (size_t iue = 0; iue < m_ue_list.size(); ++iue)
    {
      const JetData &reco = ed.reco[iue][ir];
      const ResponseHistSet &h = m_hists.resp[iue][ir];

      if (!reco.pt || !reco.pt_calib || !reco.phi || !reco.eta || !reco.e)
      {
        continue;
      }

      // one to one matching: closest reco jet not yet matched to a higher pT truth jet.
      // Jets with E <= 0 are not analysis jets (Jet-Anav3 v2/v3), so they are marked used and never matched
      m_reco_used.assign(reco.pt->size(), false);
      m_truth_match.assign(truth.pt->size(), -1);
      m_truth_closest.assign(truth.pt->size(), -1);
      m_truth_closest_dr.assign(truth.pt->size(), -1);
      for (size_t jdx = 0; jdx < reco.pt->size(); ++jdx)
      {
        if (reco.e->at(jdx) <= 0)
        {
          m_reco_used[jdx] = true;
          continue;
        }
        h.h2RecoEta_raw->Fill(reco.pt->at(jdx), reco.eta->at(jdx));
        h.h2RecoEta_calib->Fill(reco.pt_calib->at(jdx), reco.eta->at(jdx));
      }

      for (const auto &[idx, ieta] : m_truth_jets)
      {
        double pt_truth = truth.pt->at(idx);
        double eta_truth = truth.eta->at(idx);
        double phi_truth = truth.phi->at(idx);

        auto closest = closest_free_reco(eta_truth, phi_truth, reco, m_reco_used);
        if (!closest)
        {
          continue;
        }
        const auto [best_idx, dr_best] = *closest;
        const std::optional<size_t> best = best_idx;

        h.h2DeltaR->Fill(pt_truth, dr_best);
        m_truth_closest[idx] = static_cast<int>(*best);
        m_truth_closest_dr[idx] = dr_best;

        if (dr_best >= dr_max)
        {
          continue;
        }

        m_reco_used[*best] = true;
        m_truth_match[idx] = static_cast<int>(*best);
        if (m_has_truth_vertex)
        {
          h.h2TruthMatchedDz[ieta]->Fill(pt_truth, ed.zvtx - ed.zvtx_truth);
        }

        h.hTruthMatched[ieta][icent]->Fill(pt_truth);
        h.h2Resp_raw[ieta][icent]->Fill(pt_truth, reco.pt->at(*best) / pt_truth);
        h.h2Resp_calib[ieta][icent]->Fill(pt_truth, reco.pt_calib->at(*best) / pt_truth);
        h.h2DeltaEta->Fill(pt_truth, reco.eta->at(*best) - eta_truth);
      }

      if (m_dump.is_open())
      {
        dump_failed_matches(ir, iue);
      }

      if (m_has_ohcal_frac)
      {
        fill_ohcal_frac(ir, iue, icent);
      }
    }
  }
}

std::optional<std::pair<size_t, double>> JetJES::closest_free_reco(double eta, double phi, const JetData &reco, const std::vector<bool> &used)
{
  double dr_best = std::numeric_limits<double>::max();
  std::optional<size_t> best;
  for (size_t jdx = 0; jdx < reco.pt->size(); ++jdx)
  {
    if (used[jdx])
    {
      continue;
    }
    double dr = delta_r(eta, phi, reco.eta->at(jdx), reco.phi->at(jdx));
    if (dr < dr_best)
    {
      dr_best = dr;
      best = jdx;
    }
  }
  if (!best)
  {
    return std::nullopt;
  }
  return std::make_pair(*best, dr_best);
}

// OHCal fraction templates: x vs reco raw pT of the reco jets passing the data jet cuts (E > 0, |eta| < eta_max, x
// finite), for every jet and for the leading one, split by the truth match. The match is one to one over all stored truth
// jets (not only the fiducial ones of the JES), so a reco jet near the fiducial edge with its truth jet just outside it
// is still matched.
void JetJES::fill_ohcal_frac(size_t ir, size_t iue, size_t icent)
{
  const RadiusConfig &rc = m_radii[ir];
  const JetData &truth = m_event_data.truth[ir];
  const JetData &reco = m_event_data.reco[iue][ir];
  const FracArray &h = m_hists.resp[iue][ir].h2OHCalFracPt;

  if (!reco.ohcal_frac)
  {
    return;
  }

  const size_t n_reco = reco.pt->size();

  // E <= 0 jets are not analysis jets: never matched
  m_frac_reco_used.assign(n_reco, false);
  m_frac_reco_matched.assign(n_reco, false);
  for (size_t jdx = 0; jdx < n_reco; ++jdx)
  {
    m_frac_reco_used[jdx] = reco.e->at(jdx) <= 0;
  }

  // MB: no truth, every jet goes to the single match state "incl"
  if (m_has_truth)
  {
    m_frac_truth_order.resize(truth.pt->size());
    std::iota(m_frac_truth_order.begin(), m_frac_truth_order.end(), size_t{0});
    std::ranges::sort(m_frac_truth_order, [&](size_t a, size_t b)
                      { return truth.pt->at(a) > truth.pt->at(b); });

    const double dr_max = m_match_radius_frac * rc.radius;
    for (size_t idx : m_frac_truth_order)
    {
      auto closest = closest_free_reco(truth.eta->at(idx), truth.phi->at(idx), reco, m_frac_reco_used);
      if (closest && closest->second < dr_max)
      {
        m_frac_reco_used[closest->first] = true;
        m_frac_reco_matched[closest->first] = true;
      }
    }
  }
  // index into m_frac_match: matched 0 / unmatched 1, or incl 0
  auto match_index = [&](size_t jdx) -> size_t
  { return (!m_has_truth || m_frac_reco_matched[jdx]) ? 0 : 1; };

  // as Jet-Dijet: the leading jet is picked among the jets passing the cuts, then skipped if its x is not finite
  std::optional<size_t> lead;
  for (size_t jdx = 0; jdx < n_reco; ++jdx)
  {
    if (reco.e->at(jdx) <= 0 || std::abs(reco.eta->at(jdx)) >= rc.eta_max)
    {
      continue;
    }
    if (!lead || reco.pt->at(jdx) > reco.pt->at(*lead))
    {
      lead = jdx;
    }
    const double x = reco.ohcal_frac->at(jdx);
    if (std::isfinite(x))
    {
      h[0][match_index(jdx)][icent]->Fill(x, reco.pt->at(jdx));
    }
  }

  if (lead && std::isfinite(reco.ohcal_frac->at(*lead)))
  {
    h[1][match_index(*lead)][icent]->Fill(reco.ohcal_frac->at(*lead), reco.pt->at(*lead));
  }
}

void JetJES::open_dump()
{
  if (m_dump_pt_min <= 0)
  {
    return;
  }

  std::filesystem::create_directories(m_output_dir);
  std::string stem = std::filesystem::path(m_input_file).stem().string();
  std::string name = std::format("{}/Jet-JES_{}_failed_matches.txt", m_output_dir, stem);
  m_dump.open(name);
  if (!m_dump)
  {
    throw std::runtime_error(std::format("Could not open the failed-match dump {}", name));
  }

  m_dump << std::format("# Fiducial truth jets with pT >= {} GeV that fail the matching (Delta R < {} R to the closest free reco jet)\n", m_dump_pt_min, m_match_radius_frac);
  m_dump << std::format("# Input: {}\n", m_input_file);
  m_dump << "# phi in [0, 2pi); dEta, dPhi, dR: reco - truth, to the failed truth jet. Reco jets with E <= 0 are not analysis jets (never matched).\n";
  m_dump << "# The reco jet eta and the reco eta acceptance use z_MBD; the truth jets use the true vertex.\n";
  std::cout << std::format("Failed-match dump (truth pT >= {} GeV): {}\n", m_dump_pt_min, name);
}

void JetJES::dump_failed_matches(size_t ir, size_t iue)
{
  const auto &ed = m_event_data;
  const RadiusConfig &rc = m_radii[ir];
  const JetData &truth = ed.truth[ir];
  const JetData &reco = ed.reco[iue][ir];

  std::vector<size_t> failed;
  for (const auto &[idx, ieta] : m_truth_jets)
  {
    if (m_truth_match[idx] < 0 && truth.pt->at(idx) >= m_dump_pt_min)
    {
      failed.push_back(idx);
    }
  }
  if (failed.empty())
  {
    return;
  }
  m_ctr[std::format("dumped_failed_matches_{}_{}", rc.tag, m_ue_list[iue].name)] += static_cast<int>(failed.size());

  std::string file = m_chain->GetCurrentFile() ? std::filesystem::path(m_chain->GetCurrentFile()->GetName()).filename().string() : "?";
  auto &out = m_dump;
  out << std::format("\n{:=^110}\n", std::format(" FAILED MATCH | R = {:.1f} {} | {} entry {} event {} ", rc.radius, m_ue_list[iue].name, file, m_chain->GetTree() ? m_chain->GetTree()->GetReadEntry() : -1, ed.event));
  out << std::format("  global entry {} | lead truth R=0.4 pT {:.2f} GeV | z_MBD {:.2f} cm", m_entry, ed.max_pt_truth_r04, ed.zvtx);
  if (m_has_truth_vertex)
  {
    out << std::format(" | z_truth {:.2f} cm | z_MBD - z_truth {:.2f} cm", ed.zvtx_truth, ed.zvtx - ed.zvtx_truth);
  }
  else
  {
    out << " | z_truth n/a";
  }
  if (!m_is_pp)
  {
    out << std::format(" | centrality {:.1f}%", ed.centrality);
  }
  out << "\n";

  // all truth jets of the event (stored down to the Fun4All truth pT threshold, |eta| < 1.5)
  out << std::format("  Truth jets (fiducial |eta| < {}: *)\n", rc.eta_max);
  out << std::format("    {:>3} {:>8} {:>7} {:>7}  {}\n", "i", "pT", "eta", "phi", "match");
  for (size_t i = 0; i < truth.pt->size(); ++i)
  {
    bool fiducial = std::abs(truth.eta->at(i)) < rc.eta_max;
    std::string match;
    if (!fiducial)
    {
      match = "-";
    }
    else if (m_truth_match[i] >= 0)
    {
      match = std::format("reco {} (dR {:.3f})", m_truth_match[i], m_truth_closest_dr[i]);
    }
    else if (m_truth_closest[i] >= 0)
    {
      match = std::format("UNMATCHED (closest free reco {}, dR {:.3f})", m_truth_closest[i], m_truth_closest_dr[i]);
    }
    else
    {
      match = "UNMATCHED (no free reco jet)";
    }
    out << std::format("   {}{:>3} {:>8.2f} {:>7.3f} {:>7.3f}  {}\n", fiducial ? '*' : ' ', i, truth.pt->at(i), truth.eta->at(i), truth.phi->at(i), match);
  }

  // matched truth jet of each reco jet
  std::vector<int> owner(reco.pt->size(), -1);
  for (size_t i = 0; i < m_truth_match.size(); ++i)
  {
    if (m_truth_match[i] >= 0)
    {
      owner[static_cast<size_t>(m_truth_match[i])] = static_cast<int>(i);
    }
  }

  // all reco jets, with the distance to each failed truth jet
  for (size_t idx : failed)
  {
    double eta_t = truth.eta->at(idx);
    double phi_t = truth.phi->at(idx);
    out << std::format("  Reco jets vs failed truth jet {} (pT {:.2f}, eta {:.3f}, phi {:.3f}), by dR\n", idx, truth.pt->at(idx), eta_t, phi_t);
    out << std::format("    {:>3} {:>8} {:>9} {:>7} {:>7} {:>8} {:>7} {:>7} {:>7}  {}\n", "j", "pT raw", "pT calib", "eta", "phi", "E", "dEta", "dPhi", "dR", "matched to");

    std::vector<size_t> order(reco.pt->size());
    for (size_t j = 0; j < order.size(); ++j)
    {
      order[j] = j;
    }
    std::ranges::sort(order, [&](size_t a, size_t b)
                      { return delta_r(eta_t, phi_t, reco.eta->at(a), reco.phi->at(a)) < delta_r(eta_t, phi_t, reco.eta->at(b), reco.phi->at(b)); });

    for (size_t j : order)
    {
      double deta = reco.eta->at(j) - eta_t;
      double dphi = delta_phi(reco.phi->at(j), phi_t);
      std::string matched = reco.e->at(j) <= 0 ? "E <= 0 (not an analysis jet)" : (owner[j] >= 0 ? std::format("truth {}", owner[j]) : "-");
      out << std::format("    {:>3} {:>8.2f} {:>9.2f} {:>7.3f} {:>7.3f} {:>8.2f} {:>7.3f} {:>7.3f} {:>7.3f}  {}\n", j, reco.pt->at(j), reco.pt_calib->at(j), reco.eta->at(j), reco.phi->at(j), reco.e->at(j), deta, dphi, std::hypot(deta, dphi), matched);
    }
    if (reco.pt->empty())
    {
      out << "    (no reco jets above the Fun4All calibrated pT threshold in the reco eta acceptance)\n";
    }
  }
}

void JetJES::process_events()
{
  std::cout << "Processing... process_events" << std::endl;
  long long n_entries = m_chain->GetEntries();
  if (m_events_to_process)
  {
    n_entries = std::min(m_events_to_process, n_entries);
  }

  auto &ed = m_event_data;

  // Event Loop
  for (long long event = 0; event < n_entries; ++event)
  {
    // Load Event Data from TChain
    m_chain->GetEntry(event);
    m_entry = event;

    if (event % 100000 == 0)
    {
      std::cout << std::format("Processing {}/{}: {:.2f} %", event, n_entries, static_cast<double>(event) * 100. / static_cast<double>(n_entries)) << std::endl;
    }

    fill_event(EventType::ALL);
    if (m_has_truth)
    {
      m_hists.hMaxPtTruthR04_all->Fill(ed.max_pt_truth_r04);
    }

    // Event Selections, as in Jet-Anav3 (p+p: none, the single centrality class)
    auto icent = find_cent_bin(ed.centrality);
    if (!icent)
    {
      continue;
    }

    if (!m_is_pp)
    {
      fill_event(EventType::CENT);

      // OVERLAY: always; NOFLOW: only with a Calo-MBD file
      if (m_has_calo_mbd_cut)
      {
        if (!check_CaloMBD())
        {
          ++m_ctr["events_bad_calo_mbd"];
          continue;
        }
        fill_event(EventType::CALO_MBD);
      }

      // the flow selection only applies to trees made with flow modulation
      if (m_mode == Mode::OVERLAY)
      {
        if (ed.is_flow_failure_iter)
        {
          ++m_ctr["events_flow_failure_iter"];
          continue;
        }
        fill_event(EventType::FLOW_PASS_ITER);

        if (std::abs(ed.calo_v2_iter) >= m_calo_v2_max)
        {
          ++m_ctr["events_calo_v2_iter"];
          continue;
        }
        fill_event(EventType::CALO_V2);
      }
    }
    // pT-hat sample window (MB: no truth, no window)
    if (m_has_truth)
    {
      m_hists.hMaxPtTruthR04->Fill(ed.max_pt_truth_r04);
      if (!check_truth_window())
      {
        ++m_ctr["events_outside_truth_window"];
        continue;
      }
      fill_event(EventType::TRUTH_WINDOW);
      m_hists.hMaxPtTruthR04_window->Fill(ed.max_pt_truth_r04);
    }
    if (m_hists.hCentrality)
    {
      m_hists.hCentrality->Fill(ed.centrality);
    }
    if (m_has_truth_vertex)
    {
      m_hists.h2ZvtxTruth->Fill(ed.zvtx, ed.zvtx_truth);
      m_hists.h2DzLeadPt->Fill(ed.max_pt_truth_r04, ed.zvtx - ed.zvtx_truth);
    }

    // Jets
    process_jets(*icent);

    if (m_verbosity > 0)
    {
      print_event_info(event);
    }
  }

  std::cout << std::format("{:#<20}\n", "");
  std::cout << "Events Bad" << std::endl;
  for (const auto &[name, events] : m_ctr)
  {
    std::cout << std::format("{}: {}\n", name, events);
  }
  std::cout << std::format("{:#<20}\n", "");

  std::cout << "Finished... process_events" << std::endl;
}

void JetJES::print_event_info(long long event_idx) const
{
  const auto &ed = m_event_data;
  std::cout << std::format("\n{:=^70}\n", std::format(" Event {} (Entry ID: {}) ", event_idx, ed.event));
  std::cout << std::format(" Centrality: {:.2f} %, z: {:.2f} cm, Leading Truth R=0.4 Jet pT: {:.2f} GeV\n", ed.centrality, ed.zvtx, ed.max_pt_truth_r04);

  auto print_jets = [this](std::string_view label, const JetData &jd)
  {
    size_t njets = jd.pt ? jd.pt->size() : 0;
    std::cout << std::format("  [{}] N_jets: {}\n", label, njets);

    if (m_verbosity >= 2)
    {
      for (size_t i = 0; i < njets; ++i)
      {
        double pt_cal = jd.pt_calib ? jd.pt_calib->at(i) : 0.0;
        std::cout << std::format("    {:>4} | pT: {:>8.2f} | pT calib: {:>8.2f} | eta: {:>7.3f} | phi: {:>7.3f}\n", i, jd.pt->at(i), pt_cal, jd.eta->at(i), jd.phi->at(i));
      }
    }
  };

  for (size_t ir = 0; ir < m_radii.size(); ++ir)
  {
    print_jets(std::format("{}_truth", m_radii[ir].tag), ed.truth[ir]);
    for (size_t iue = 0; iue < m_ue_list.size(); ++iue)
    {
      print_jets(std::format("{}_{}", m_radii[ir].tag, m_ue_list[iue].name), ed.reco[iue][ir]);
    }
  }
  std::cout << std::format("{:=^70}\n", "");
}

void JetJES::save_results() const
{
  std::filesystem::create_directories(m_output_dir);

  std::filesystem::path input_path(m_input_file);
  std::string output_stem = input_path.stem().string();
  std::string output_filename = std::format("{}/Jet-JES_{}.root", m_output_dir, output_stem);

  auto output_file = std::make_unique<TFile>(output_filename.c_str(), "RECREATE");

  auto save_all = [](auto &...collections)
  {
    auto save = [](const auto &container)
    {
      for (const auto &[name, hist] : container)
      {
        if (hist)
        {
          hist->Write();
        }
      }
    };
    (save(collections), ...);  // Fold expression
  };

  save_all(m_hists1D, m_hists2D);

  output_file->Close();

  std::cout << std::format("Results saved to: {}", output_filename) << std::endl;
}

// ====================================================================
// Main function remains clean and simple
// ====================================================================
int main(int argc, const char *const argv[])
{
  gROOT->SetBatch(true);
  TH1::AddDirectory(false);

  if (argc < 3 || argc > 11)
  {
    std::cout << "Usage: " << argv[0] << " input_file calo_mbd_file [events] [output_directory] [lead_truth_pt_min] [lead_truth_pt_max] [sigma_cut] [verbosity] [mode] [dump_pt_min]" << std::endl;
    std::cout << "  mode: 0 data+MC overlay made with flow (iter / iternf, Calo-MBD file required; default)" << std::endl;
    std::cout << "        1 p+p MC (Fun4All_PP_JES; unsub / iternf, calo_mbd_file not used: pass none)" << std::endl;
    std::cout << "        2 Au+Au without flow (HIJING embedded MC or no-flow overlay; iternf; calo_mbd_file optional: none = no Calo-MBD cut)" << std::endl;
    std::cout << "        3 minimum-bias HIJING without truth (iternf; no window, no matching, OHCal fraction templates only; calo_mbd_file optional)" << std::endl;
    std::cout << "  dump_pt_min: write the fiducial truth jets with pT >= dump_pt_min [GeV] that fail the matching to a text file (<= 0: off, default)" << std::endl;
    return 1;
  }

  int ctr = 1;
  const std::string input_file = argv[ctr++];
  const std::string calo_mbd_file = argv[ctr++];
  long long events = (argc >= ctr + 1) ? std::atoll(argv[ctr++]) : 0;
  std::string output_dir = (argc >= ctr + 1) ? argv[ctr++] : ".";
  double lead_truth_pt_min = (argc >= ctr + 1) ? std::stod(argv[ctr++]) : 0;
  double lead_truth_pt_max = (argc >= ctr + 1) ? std::stod(argv[ctr++]) : -1;
  double sigma_cut = (argc >= ctr + 1) ? std::stod(argv[ctr++]) : 3.5;
  int verbosity = (argc >= ctr + 1) ? std::atoi(argv[ctr++]) : 0;
  int mode_value = (argc >= ctr + 1) ? std::atoi(argv[ctr++]) : 0;
  double dump_pt_min = (argc >= ctr + 1) ? std::stod(argv[ctr++]) : -1;
  if (mode_value < 0 || mode_value > 3)
  {
    std::cout << std::format("Invalid mode {}: expected 0 (overlay), 1 (p+p), 2 (Au+Au without flow) or 3 (MB HIJING without truth)", mode_value) << std::endl;
    return 1;
  }
  const auto mode = static_cast<JetJES::Mode>(mode_value);
  const bool use_calo_mbd = JetJES::uses_calo_mbd(mode, calo_mbd_file);

  std::cout << std::format("{:#<20}\n", "");
  std::cout << std::format("Run Params\n");
  std::cout << std::format("Mode: {} ({})\n", mode_value, JetJES::mode_name(mode));
  std::cout << std::format("Input: {}\n", input_file);
  std::cout << std::format("Calo-MBD File: {}\n", use_calo_mbd ? calo_mbd_file : "not used");
  std::cout << std::format("Events: {}\n", events);
  std::cout << std::format("Output Dir: {}\n", output_dir);
  std::cout << std::format("Lead Truth R=0.4 Jet pT Window: [{}, {}) [GeV]\n", lead_truth_pt_min, lead_truth_pt_max < 0 ? "inf" : std::format("{}", lead_truth_pt_max));
  if (use_calo_mbd)
  {
    std::cout << std::format("Sigma Cut: {:.2f}\n", sigma_cut);
  }
  std::cout << std::format("Verbosity: {}\n", verbosity);
  std::cout << std::format("Failed-Match Dump: {}\n", dump_pt_min > 0 ? std::format("truth pT >= {} GeV", dump_pt_min) : "off");
  std::cout << std::format("{:#<20}\n", "");

  try
  {
    JetJES analysis(input_file, calo_mbd_file, events, output_dir, mode);
    analysis.set_lead_truth_pt_min(lead_truth_pt_min);
    analysis.set_lead_truth_pt_max(lead_truth_pt_max);
    analysis.set_sigma_cut(sigma_cut);
    analysis.set_verbosity(verbosity);
    analysis.set_dump_pt_min(dump_pt_min);
    analysis.run();
  }
  catch (const std::exception &e)
  {
    std::cout << std::format("An exception occurred: {}", e.what()) << std::endl;
    return 1;
  }

  std::cout << "Analysis complete." << std::endl;
  return 0;
}
