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
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numbers>
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
// Jet energy scale in data+MC overlay: matches the truth jets to the reconstructed (iter / iternf) jets
// and fills the response pT_reco / pT_truth vs pT_truth in the eta bins of the legacy eta dependent JES.
// One pT-hat sample per job: the event is kept if the leading truth R=0.4 jet pT is in the sample's
// window, and the samples are weighted (1 / IntegratedLumi_job of h_gen_integral) and summed afterwards.
class JetJES
{
 public:
  JetJES(std::string input_file, std::string calo_mbd_file, long long events, std::string output_dir)
    : m_input_file(std::move(input_file))
    , m_calo_mbd_file(std::move(calo_mbd_file))
    , m_events_to_process(events)
    , m_output_dir(std::move(output_dir))
  {
  }

  void run()
  {
    setup_calo_mbd_cut();
    setup_chain();
    init_hists();
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

 private:
  static constexpr double m_cent_high = 59.5;

  static constexpr std::array<double, 7> m_cent_bins = {0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0};
  static constexpr size_t m_num_cent_bins = m_cent_bins.size() - 1;

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

  // UE: iter -> Iterative Method, iternf -> Iterative Method w/o flow modulation
  static constexpr std::array<std::string_view, 2> m_ue_list = {"iter", "iternf"};

  // Matching radius: Delta R < m_match_radius_frac * R
  static constexpr double m_match_radius_frac = 0.75;

  // Jet-Anav3 v3 selection: |calo v2| < m_calo_v2_max (iter, also for iternf) and reco jet energy > 0
  static constexpr float m_calo_v2_max = 0.48F;

  template <typename T>
  using EtaCentArray = std::array<std::array<T *, m_num_cent_bins>, m_num_eta_bins>;

  struct ResponseHistSet
  {
    EtaCentArray<TH2> h2Resp_raw{};
    EtaCentArray<TH2> h2Resp_calib{};
    EtaCentArray<TH1> hTruthMatched{};
    TH2 *h2DeltaEta{nullptr};
    TH2 *h2DeltaR{nullptr};
    // all reco jets, as h2JetEta in Jet-Anav3
    TH2 *h2RecoEta_raw{nullptr};
    TH2 *h2RecoEta_calib{nullptr};
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
    // [ue][radius]
    std::array<std::array<ResponseHistSet, m_radii.size()>, m_ue_list.size()> resp{};
  };

  AnalysisHists m_hists;

  struct JetData
  {
    std::vector<double> *pt{nullptr};        // NOLINT(misc-non-private-member-variables-in-classes)
    std::vector<double> *pt_calib{nullptr};  // NOLINT(misc-non-private-member-variables-in-classes)
    std::vector<double> *phi{nullptr};       // NOLINT(misc-non-private-member-variables-in-classes)
    std::vector<double> *eta{nullptr};       // NOLINT(misc-non-private-member-variables-in-classes)
    std::vector<double> *e{nullptr};         // NOLINT(misc-non-private-member-variables-in-classes)
  };

  struct EventData
  {
    friend class JetJES;

   private:
    int event{0};
    double zvtx{0.0};
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
    std::array<std::array<JetData, m_radii.size()>, m_ue_list.size()> reco;
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

  // Hists
  std::map<std::string, std::unique_ptr<TH1>> m_hists1D;
  std::map<std::string, std::unique_ptr<TH2>> m_hists2D;

  double m_sigma_cut{3.5};
  bool m_has_calo_mbd_cut{false};
  std::unique_ptr<TProfile> m_profile_calo_mbd{nullptr};

  // per event buffers
  std::vector<TruthJet> m_truth_jets;
  std::vector<bool> m_reco_used;

  // --- Private Helper Methods ---
  void setup_chain();
  void setup_calo_mbd_cut();
  void init_hists();

  bool check_CaloMBD() const;
  bool check_truth_window() const;
  static std::optional<size_t> find_cent_bin(double cent);
  static std::optional<size_t> find_eta_bin(double eta, const RadiusConfig &rc);
  static double delta_r(double eta1, double phi1, double eta2, double phi2);

  void process_jets(size_t icent);
  void process_events();
  void print_event_info(long long event_idx) const;

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
  set_branch("centrality", &ed.centrality);
  set_branch("emcal_energy", &ed.emcal_energy);
  set_branch("ihcal_energy", &ed.ihcal_energy);
  set_branch("ohcal_energy", &ed.ohcal_energy);
  set_branch("mbd_charge_south", &ed.mbd_charge_south);
  set_branch("mbd_charge_north", &ed.mbd_charge_north);
  set_branch("is_flow_failure_iter", &ed.is_flow_failure_iter);
  set_branch("calo_v2_iter", &ed.calo_v2_iter);
  set_branch("max_pt_truth_r04", &ed.max_pt_truth_r04);

  for (size_t ir = 0; ir < m_radii.size(); ++ir)
  {
    std::string_view r = m_radii[ir].tag;

    JetData &truth = ed.truth[ir];
    set_branch(std::format("pt_truth_{}", r), &truth.pt);
    set_branch(std::format("phi_truth_{}", r), &truth.phi);
    set_branch(std::format("eta_truth_{}", r), &truth.eta);

    for (size_t iue = 0; iue < m_ue_list.size(); ++iue)
    {
      std::string_view ue = m_ue_list[iue];
      JetData &reco = ed.reco[iue][ir];
      set_branch(std::format("pt_{}_{}", ue, r), &reco.pt);
      set_branch(std::format("pt_calib_{}_{}", ue, r), &reco.pt_calib);
      set_branch(std::format("phi_{}_{}", ue, r), &reco.phi);
      set_branch(std::format("eta_{}_{}", ue, r), &reco.eta);
      set_branch(std::format("e_{}_{}", ue, r), &reco.e);
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

  std::vector<std::string> event_labels = {"All", "Centrality 0-60%", "Good Calo-MBD", "Flow Pass Iter", std::format("|Calo v_{{2}}| < {}", m_calo_v2_max),
                                           std::format("Lead Truth R=0.4 p_{{T}} Window [{}, {})", m_lead_truth_pt_min,
                                                       m_lead_truth_pt_max < 0 ? "#infty" : std::format("{}", m_lead_truth_pt_max))};

  int bins_event = static_cast<int>(event_labels.size());
  m_hists1D["hEvent"] = std::make_unique<TH1F>("hEvent", "Event Type; Type; Events", bins_event, 0, bins_event);
  for (size_t i = 0; i < event_labels.size(); ++i)
  {
    m_hists1D["hEvent"]->GetXaxis()->SetBinLabel(static_cast<int>(i + 1), event_labels[i].c_str());
  }

  m_hists1D["hCentrality"] = std::make_unique<TH1F>("hCentrality", "Selected Events; Centrality [%]; Events", 60, -0.5, 59.5);

  m_hists1D["hMaxPtTruthR04_all"] = std::make_unique<TH1F>("hMaxPtTruthR04_all", "All Events; Leading Truth R=0.4 Jet p_{T} [GeV]; Events", bins_lead_pt, lead_pt_low, lead_pt_high);
  m_hists1D["hMaxPtTruthR04"] = std::make_unique<TH1F>("hMaxPtTruthR04", "Event Selection (before window); Leading Truth R=0.4 Jet p_{T} [GeV]; Events", bins_lead_pt, lead_pt_low, lead_pt_high);
  m_hists1D["hMaxPtTruthR04_window"] = std::make_unique<TH1F>("hMaxPtTruthR04_window", "Event Selection + Window; Leading Truth R=0.4 Jet p_{T} [GeV]; Events", bins_lead_pt, lead_pt_low, lead_pt_high);

  m_hists.hEvent = m_hists1D["hEvent"].get();
  m_hists.hCentrality = m_hists1D["hCentrality"].get();
  m_hists.hMaxPtTruthR04_all = m_hists1D["hMaxPtTruthR04_all"].get();
  m_hists.hMaxPtTruthR04 = m_hists1D["hMaxPtTruthR04"].get();
  m_hists.hMaxPtTruthR04_window = m_hists1D["hMaxPtTruthR04_window"].get();

  for (size_t ir = 0; ir < m_radii.size(); ++ir)
  {
    const RadiusConfig &rc = m_radii[ir];
    std::string_view r = rc.tag;

    std::string truth_eta_name = std::format("h2TruthEta_{}", r);
    m_hists2D[truth_eta_name] = std::make_unique<TH2F>(truth_eta_name.c_str(), std::format("R = {:.1f}, All Truth Jets; p_{{T}}^{{truth}} [GeV]; #eta^{{truth}}", rc.radius).c_str(), bins_pt_eta2D, pt_eta2D_low, pt_eta2D_high, bins_eta, eta_low, eta_high);
    m_hists.h2TruthEta[ir] = m_hists2D[truth_eta_name].get();

    for (size_t ieta = 0; ieta < m_num_eta_bins; ++ieta)
    {
      for (size_t icent = 0; icent < m_num_cent_bins; ++icent)
      {
        int cent_low = static_cast<int>(m_cent_bins[icent]);
        int cent_high = static_cast<int>(m_cent_bins[icent + 1]);
        std::string suffix = std::format("eta{}_cent_{}_{}", ieta, cent_low, cent_high);
        std::string label = std::format("R = {:.1f}, {:.2f} #leq #eta^{{truth}} < {:.2f}, {} - {}%", rc.radius, rc.eta_edges[ieta], rc.eta_edges[ieta + 1], cent_low, cent_high);

        std::string truth_name = std::format("hTruth_{}_{}", r, suffix);
        m_hists1D[truth_name] = std::make_unique<TH1F>(truth_name.c_str(), std::format("{}; p_{{T}}^{{truth}} [GeV]; Jets", label).c_str(), bins_pt, pt_low, pt_high);
        m_hists.hTruth[ir][ieta][icent] = m_hists1D[truth_name].get();

        for (size_t iue = 0; iue < m_ue_list.size(); ++iue)
        {
          std::string_view ue = m_ue_list[iue];
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
      std::string_view ue = m_ue_list[iue];
      ResponseHistSet &h = m_hists.resp[iue][ir];

      std::string deta_name = std::format("h2DeltaEta_{}_{}", r, ue);
      m_hists2D[deta_name] = std::make_unique<TH2F>(deta_name.c_str(), std::format("R = {:.1f}, {}, Matched; p_{{T}}^{{truth}} [GeV]; #eta^{{reco}} - #eta^{{truth}}", rc.radius, ue).c_str(), bins_pt, pt_low, pt_high, bins_deta, deta_low, deta_high);
      h.h2DeltaEta = m_hists2D[deta_name].get();

      std::string dr_name = std::format("h2DeltaR_{}_{}", r, ue);
      m_hists2D[dr_name] = std::make_unique<TH2F>(dr_name.c_str(), std::format("R = {:.1f}, {}, Closest Reco Jet (before #DeltaR cut); p_{{T}}^{{truth}} [GeV]; #DeltaR", rc.radius, ue).c_str(), bins_pt, pt_low, pt_high, bins_dr, dr_low, dr_high);
      h.h2DeltaR = m_hists2D[dr_name].get();

      for (const char *calib : {"raw", "calib"})
      {
        std::string reco_eta_name = std::format("h2RecoEta_{}_{}_{}", calib, r, ue);
        m_hists2D[reco_eta_name] = std::make_unique<TH2F>(reco_eta_name.c_str(), std::format("R = {:.1f}, {}, All Reco Jets; p_{{T}}^{{reco, {}}} [GeV]; #eta^{{reco}}", rc.radius, ue, calib).c_str(), bins_pt_eta2D, pt_eta2D_low, pt_eta2D_high, bins_eta, eta_low, eta_high);
        (std::string_view(calib) == "raw" ? h.h2RecoEta_raw : h.h2RecoEta_calib) = m_hists2D[reco_eta_name].get();
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

std::optional<size_t> JetJES::find_cent_bin(double cent)
{
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

double JetJES::delta_r(double eta1, double phi1, double eta2, double phi2)
{
  double dphi = std::abs(phi1 - phi2);
  if (dphi > std::numbers::pi)
  {
    dphi = 2 * std::numbers::pi - dphi;
  }
  return std::hypot(eta1 - eta2, dphi);
}

void JetJES::process_jets(size_t icent)
{
  const auto &ed = m_event_data;

  for (size_t ir = 0; ir < m_radii.size(); ++ir)
  {
    const RadiusConfig &rc = m_radii[ir];
    const JetData &truth = ed.truth[ir];

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

        double dr_best = std::numeric_limits<double>::max();
        std::optional<size_t> best;
        for (size_t jdx = 0; jdx < reco.pt->size(); ++jdx)
        {
          if (m_reco_used[jdx])
          {
            continue;
          }
          double dr = delta_r(eta_truth, phi_truth, reco.eta->at(jdx), reco.phi->at(jdx));
          if (dr < dr_best)
          {
            dr_best = dr;
            best = jdx;
          }
        }

        if (!best)
        {
          continue;
        }

        h.h2DeltaR->Fill(pt_truth, dr_best);

        if (dr_best >= dr_max)
        {
          continue;
        }

        m_reco_used[*best] = true;

        h.hTruthMatched[ieta][icent]->Fill(pt_truth);
        h.h2Resp_raw[ieta][icent]->Fill(pt_truth, reco.pt->at(*best) / pt_truth);
        h.h2Resp_calib[ieta][icent]->Fill(pt_truth, reco.pt_calib->at(*best) / pt_truth);
        h.h2DeltaEta->Fill(pt_truth, reco.eta->at(*best) - eta_truth);
      }
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

    if (event % 100000 == 0)
    {
      std::cout << std::format("Processing {}/{}: {:.2f} %", event, n_entries, static_cast<double>(event) * 100. / static_cast<double>(n_entries)) << std::endl;
    }

    m_hists.hEvent->Fill(static_cast<std::uint8_t>(EventType::ALL));
    m_hists.hMaxPtTruthR04_all->Fill(ed.max_pt_truth_r04);

    // Event Selections, as in Jet-Anav3
    auto icent = find_cent_bin(ed.centrality);
    if (!icent)
    {
      continue;
    }
    m_hists.hEvent->Fill(static_cast<std::uint8_t>(EventType::CENT));

    if (!check_CaloMBD())
    {
      ++m_ctr["events_bad_calo_mbd"];
      continue;
    }
    m_hists.hEvent->Fill(static_cast<std::uint8_t>(EventType::CALO_MBD));

    if (ed.is_flow_failure_iter)
    {
      ++m_ctr["events_flow_failure_iter"];
      continue;
    }
    m_hists.hEvent->Fill(static_cast<std::uint8_t>(EventType::FLOW_PASS_ITER));

    if (std::abs(ed.calo_v2_iter) >= m_calo_v2_max)
    {
      ++m_ctr["events_calo_v2_iter"];
      continue;
    }
    m_hists.hEvent->Fill(static_cast<std::uint8_t>(EventType::CALO_V2));
    m_hists.hMaxPtTruthR04->Fill(ed.max_pt_truth_r04);

    // pT-hat sample window
    if (!check_truth_window())
    {
      ++m_ctr["events_outside_truth_window"];
      continue;
    }
    m_hists.hEvent->Fill(static_cast<std::uint8_t>(EventType::TRUTH_WINDOW));
    m_hists.hMaxPtTruthR04_window->Fill(ed.max_pt_truth_r04);
    m_hists.hCentrality->Fill(ed.centrality);

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
      print_jets(std::format("{}_{}", m_radii[ir].tag, m_ue_list[iue]), ed.reco[iue][ir]);
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

  if (argc < 3 || argc > 9)
  {
    std::cout << "Usage: " << argv[0] << " input_file calo_mbd_file [events] [output_directory] [lead_truth_pt_min] [lead_truth_pt_max] [sigma_cut] [verbosity]" << std::endl;
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

  std::cout << std::format("{:#<20}\n", "");
  std::cout << std::format("Run Params\n");
  std::cout << std::format("Input: {}\n", input_file);
  std::cout << std::format("Calo-MBD File: {}\n", calo_mbd_file);
  std::cout << std::format("Events: {}\n", events);
  std::cout << std::format("Output Dir: {}\n", output_dir);
  std::cout << std::format("Lead Truth R=0.4 Jet pT Window: [{}, {}) [GeV]\n", lead_truth_pt_min, lead_truth_pt_max < 0 ? "inf" : std::format("{}", lead_truth_pt_max));
  std::cout << std::format("Sigma Cut: {:.2f}\n", sigma_cut);
  std::cout << std::format("Verbosity: {}\n", verbosity);
  std::cout << std::format("{:#<20}\n", "");

  try
  {
    JetJES analysis(input_file, calo_mbd_file, events, output_dir);
    analysis.set_lead_truth_pt_min(lead_truth_pt_min);
    analysis.set_lead_truth_pt_max(lead_truth_pt_max);
    analysis.set_sigma_cut(sigma_cut);
    analysis.set_verbosity(verbosity);
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
