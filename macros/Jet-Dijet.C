// Clean dijet control sample (tasks/jet-dijet-control.md)
//
// Reads only the stored 'iter' jets (the no-flow iternf jets) for R = 0.2 and 0.3, with the event selection of
// Jet-Anav3 (|z| < 10 cm and MB from the skim, Good Calo-Cent, centrality 0 - 60 %). Per radius and 10 % centrality
// class and raw pT_lead class it fills TH2s of the subleading raw pT (and of x_J) vs delta phi, so that the subleading
// pT, x_J and delta phi cuts can be scanned offline.
//
// With the chosen cut per radius (pT_sub >= pt_sub_min, delta phi > 7pi/8) it also fills, for the leading jet, TH2s of
// jet time - MBD t0 vs OHCal energy fraction in a grid of 10 % centrality classes x raw pT_lead bins, for the dijet
// sample and for the inclusive comparison (leading jet raw pT >= pt_sub_min, no dijet requirement), and the raw pT x
// centrality of every jet, in dijet events and in all events.

// -- utils
#include "myUtils.C"

// ====================================================================
// Standard C++ Includes
// ====================================================================
#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <format>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numbers>
#include <stdexcept>
#include <string>
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
class JetDijet
{
 public:
  JetDijet(std::string input_file, std::string calo_mbd_file, long long events, std::string output_dir)
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

  void set_jet_pt_raw_min(double pt) { m_jet_pt_raw_min = pt; }
  void set_sigma_cut(double sigma) { m_sigma_cut = sigma; }
  void set_verbosity(int verbosity) { m_verbosity = verbosity; }
  // Dijet selection for the timing and jet pT hists: subleading raw pT >= pt_sub_min of the radius (the 0-20 % cut of
  // the purity scan, used in every centrality class)
  void set_pt_sub_min(double r02, double r03) { m_pt_sub_min = {r02, r03}; }

 private:
  // Centrality: the selection of Jet-Anav3 (centrality <= 59.5) and 10 % classes [lo, hi), the last one closed
  static constexpr double m_cent_high = 59.5;
  static constexpr std::array<double, 7> m_cent_edges = {0, 10, 20, 30, 40, 50, 60};
  static constexpr size_t m_n_cent = m_cent_edges.size() - 1;

  // Radii: R = 0.2, 0.3 with the jet |eta| cut of Jet-Anav3
  static constexpr size_t m_n_radii = 2;
  static constexpr std::array<const char *, m_n_radii> m_radius_tag = {"r02", "r03"};
  static constexpr std::array<double, m_n_radii> m_jet_eta_max = {0.9, 0.8};

  // Raw pT_sub binning [GeV] of the pT_sub x delta phi hists, fine so the purity scan can place the pT_sub cut between
  // whole GeV: 0.25 GeV from 7 to 30 GeV, 1 GeV to 40 GeV, then 42.5, 45, 50, 55, 60, 70, 80, 100 GeV.
  static constexpr size_t m_n_sub_fine = 92;  // (30 - 7) / 0.25
  static constexpr size_t m_n_sub_mid = 10;   // 31 ... 40
  static constexpr std::array<double, 8> m_sub_tail = {42.5, 45, 50, 55, 60, 70, 80, 100};
  static constexpr std::array<double, 1 + m_n_sub_fine + m_n_sub_mid + 8> m_sub_edges = []
  {
    std::array<double, 1 + m_n_sub_fine + m_n_sub_mid + 8> e{};
    size_t k = 0;
    for (size_t i = 0; i <= m_n_sub_fine; ++i)
    {
      e[k++] = 7.0 + 0.25 * static_cast<double>(i);  // exact in binary
    }
    for (size_t i = 1; i <= m_n_sub_mid; ++i)
    {
      e[k++] = 30.0 + static_cast<double>(i);
    }
    for (double x : m_sub_tail)
    {
      e[k++] = x;
    }
    return e;
  }();
  // Raw pT_lead classes [GeV] of the pT_sub x delta phi and x_J x delta phi hists (15, 20, 30, 40: the timing ranges)
  static constexpr std::array<int, 9> m_lead_edges = {7, 10, 15, 20, 25, 30, 40, 60, 100};
  static constexpr size_t m_n_lead = m_lead_edges.size() - 1;
  // delta phi: pi / 72 wide, so pi/3, pi/2, 2pi/3, 3pi/4 and 7pi/8 are bin edges
  static constexpr int m_bins_dphi = 72;
  static constexpr int m_bins_xj = 20;
  static constexpr double m_dphi_b2b = 7 * std::numbers::pi / 8;  // back-to-back cut of the dijet selection (the plot script uses the same)

  // Leading jet timing hists, binning as h2LeadJetTimeOHCalFrac of Jet-Anav3: OHCal fraction 60 bins over [-0.25, 1.25],
  // jet t - MBD t0 100 bins over +-25 ns
  static constexpr int m_bins_frac = 60;
  static constexpr double m_frac_low = -0.25;
  static constexpr double m_frac_high = 1.25;
  static constexpr int m_bins_dt = 100;
  static constexpr double m_dt_max = 25; /*ns*/
  // Raw pT_lead bins of the timing grid [GeV]: from pt_sub_min of the radius, then these edges above it
  static constexpr std::array<double, 6> m_time_lead_upper = {20, 25, 30, 40, 60, 100};
  // Jet raw pT x centrality: 1 GeV over [0, 300], 1 % centrality bins
  static constexpr int m_bins_jet_pt = 300;
  static constexpr double m_jet_pt_high = 300;
  // Selections of the timing and jet pT hists
  enum Sel : std::uint8_t
  {
    SEL_DIJET,  // leading + subleading, delta phi > 7pi/8, pT_sub >= pt_sub_min
    SEL_INCL,   // timing: leading jet pT >= pt_sub_min, no dijet requirement; jet pT: every selected event
    N_SEL
  };
  static constexpr std::array<const char *, N_SEL> m_sel_time_tag = {"dijet", "incl"};
  static constexpr std::array<const char *, N_SEL> m_sel_jet_tag = {"dijet", "all"};

  struct JetData
  {
    std::vector<double> *pt{nullptr};          // NOLINT(misc-non-private-member-variables-in-classes)
    std::vector<double> *e{nullptr};           // NOLINT(misc-non-private-member-variables-in-classes)
    std::vector<double> *phi{nullptr};         // NOLINT(misc-non-private-member-variables-in-classes)
    std::vector<double> *eta{nullptr};         // NOLINT(misc-non-private-member-variables-in-classes)
    std::vector<double> *t{nullptr};           // NOLINT(misc-non-private-member-variables-in-classes)
    std::vector<double> *ohcal_frac{nullptr};  // NOLINT(misc-non-private-member-variables-in-classes)
  };

  struct EventData
  {
    int run{0};
    int event{0};
    double centrality{0};
    double emcal_energy{0};
    double ihcal_energy{0};
    double ohcal_energy{0};
    double mbd_charge_south{0};
    double mbd_charge_north{0};
    float t0{0};
    std::array<JetData, m_n_radii> jets;
  };

  struct DijetHists
  {
    TH2 *h2NjetsCent{nullptr};
    // [centrality class][pT_lead class]
    std::array<std::array<TH2 *, m_n_lead>, m_n_cent> h2PtSubDphi{};
    std::array<std::array<TH2 *, m_n_lead>, m_n_cent> h2XjDphi{};

    // Leading jet timing, [selection][centrality class][pT_lead bin]: jet t - MBD t0 vs OHCal fraction
    std::vector<double> time_lead_edges;  // pt_sub_min, then the m_time_lead_upper edges above it
    std::array<std::array<std::vector<TH2 *>, m_n_cent>, N_SEL> h2LeadTimeOHCalFrac{};
    std::array<TH1 *, N_SEL> hLeadTimingStatus{};
    // Raw pT x centrality of every jet passing the jet cuts, [selection]: in dijet events, in all selected events
    std::array<TH2 *, N_SEL> h2JetPtCent{};
    // Raw pT x centrality of the leading jet, [selection]: dijet sample, inclusive (pT_lead >= pt_sub_min)
    std::array<TH2 *, N_SEL> h2LeadPtCent{};
  };

  enum EventBin : std::uint8_t
  {
    ENTRIES,
    CALOCENT,
    CENT,
    LEAD_R02,
    DIJET_R02,
    LEAD_R03,
    DIJET_R03,
    N_EVENT_BINS
  };

  // --- Configuration ---
  std::string m_input_file;
  std::string m_calo_mbd_file;
  long long m_events_to_process;
  std::string m_output_dir;
  int m_verbosity{0};
  double m_jet_pt_raw_min{7}; /*GeV*/
  double m_sigma_cut{3.5};
  std::array<double, m_n_radii> m_pt_sub_min{14.25, 21.5}; /*GeV, raw*/

  // --- State ---
  EventData m_event_data;
  std::unique_ptr<TChain> m_chain;
  std::unique_ptr<TProfile> m_profile_calo_mbd;

  TH1 *m_hEvent{nullptr};
  TH1 *m_hCentrality{nullptr};
  std::array<DijetHists, m_n_radii> m_hists;

  std::map<std::string, std::unique_ptr<TH1>> m_hists_all;  // owns every hist, saved by name

  // --- Helpers ---
  void setup_calo_mbd_cut();
  void setup_chain();
  void init_hists();
  bool check_CaloMBD() const;
  static int cent_class(double cent);
  static std::string edge_tag(double x);
  bool pass_jet(const JetData &jd, size_t idx, size_t ir) const;
  void fill_lead_timing(size_t ir, Sel sel, size_t ic, size_t il);
  void process_dijet(size_t ir, int cls);
  void process_events();
  void save_results() const;

  template <typename T, typename... Args>
  T *book(const std::string &name, Args &&...args)
  {
    auto h = std::make_unique<T>(name.c_str(), std::forward<Args>(args)...);
    T *ptr = h.get();
    m_hists_all[name] = std::move(h);
    return ptr;
  }
};

// ====================================================================
// Implementation
// ====================================================================

// The Calo-MBD consistency cut, as in Jet-Anav3: total calo energy within sigma_cut standard deviations of its mean at
// the event's MBD charge, from h2TotalCaloE_MBD of the Event QA file
void JetDijet::setup_calo_mbd_cut()
{
  if (m_calo_mbd_file.empty() || !std::filesystem::exists(m_calo_mbd_file))
  {
    throw std::runtime_error(std::format("Calo-MBD file does not exist: '{}'", m_calo_mbd_file));
  }

  std::unique_ptr<TFile> tfile(TFile::Open(m_calo_mbd_file.c_str(), "READ"));
  if (!tfile || tfile->IsZombie())
  {
    throw std::runtime_error(std::format("Could not open Calo-MBD file: {}", m_calo_mbd_file));
  }

  auto *h2 = dynamic_cast<TH2 *>(tfile->Get("h2TotalCaloE_MBD"));
  if (!h2)
  {
    throw std::runtime_error(std::format("h2TotalCaloE_MBD not found in {}", m_calo_mbd_file));
  }

  TProfile *pfx = h2->ProfileX("pfx_calo_mbd_cut", 1, -1, "s");
  pfx->SetDirectory(nullptr);
  m_profile_calo_mbd.reset(pfx);

  std::cout << std::format("Loaded h2TotalCaloE_MBD from {} (sigma cut: {:.2f})\n", m_calo_mbd_file, m_sigma_cut);
}

bool JetDijet::check_CaloMBD() const
{
  const auto &ed = m_event_data;
  const double total_energy = ed.emcal_energy + ed.ihcal_energy + ed.ohcal_energy;
  const double mbd_total = ed.mbd_charge_south + ed.mbd_charge_north;

  const int bin = m_profile_calo_mbd->FindBin(mbd_total);
  if (bin < 1 || bin > m_profile_calo_mbd->GetNbinsX())
  {
    return false;
  }
  const double mean = m_profile_calo_mbd->GetBinContent(bin);
  const double sigma = m_profile_calo_mbd->GetBinError(bin);
  return sigma > 0.0 && std::abs(total_energy - mean) <= m_sigma_cut * sigma;
}

void JetDijet::setup_chain()
{
  std::cout << "Processing... setup_chain" << std::endl;

  m_chain = myUtils::setupTChain(m_input_file, "T");
  if (m_chain == nullptr)
  {
    throw std::runtime_error(std::format("Error in TChain Setup from file: {}", m_input_file));
  }

  m_chain->SetBranchStatus("*", false);

  auto enable = [this](const std::string &name, void *address)
  {
    if (!m_chain->GetBranch(name.c_str()))
    {
      throw std::runtime_error(std::format("Could not find Branch '{}' in file '{}'", name, m_input_file));
    }
    m_chain->SetBranchStatus(name.c_str(), true);
    m_chain->SetBranchAddress(name.c_str(), address);
  };

  auto &ed = m_event_data;
  enable("run", &ed.run);
  enable("event", &ed.event);
  enable("centrality", &ed.centrality);
  enable("emcal_energy", &ed.emcal_energy);
  enable("ihcal_energy", &ed.ihcal_energy);
  enable("ohcal_energy", &ed.ohcal_energy);
  enable("mbd_charge_south", &ed.mbd_charge_south);
  enable("mbd_charge_north", &ed.mbd_charge_north);
  enable("t0", &ed.t0);

  // the stored 'iter' jets are the no-flow (iternf) jets
  for (size_t ir = 0; ir < m_n_radii; ++ir)
  {
    const char *r = m_radius_tag[ir];
    JetData &jd = ed.jets[ir];
    enable(std::format("pt_iter_{}", r), &jd.pt);
    enable(std::format("e_iter_{}", r), &jd.e);
    enable(std::format("phi_iter_{}", r), &jd.phi);
    enable(std::format("eta_iter_{}", r), &jd.eta);
    enable(std::format("t_iter_{}", r), &jd.t);
    enable(std::format("ohcal_frac_iter_{}", r), &jd.ohcal_frac);
  }
}

void JetDijet::init_hists()
{
  std::cout << "Processing... init_hists" << std::endl;

  // fixed-width axes as edge arrays for the variable-binning TH2 constructor
  std::vector<double> dphi_edges(m_bins_dphi + 1);
  for (int i = 0; i <= m_bins_dphi; ++i)
  {
    dphi_edges[static_cast<size_t>(i)] = std::numbers::pi * i / m_bins_dphi;
  }
  std::vector<double> xj_edges(m_bins_xj + 1);
  for (int i = 0; i <= m_bins_xj; ++i)
  {
    xj_edges[static_cast<size_t>(i)] = static_cast<double>(i) / m_bins_xj;
  }

  m_hEvent = book<TH1D>("hEvent", "Events; ; Events", N_EVENT_BINS, 0, N_EVENT_BINS);
  const std::array<std::string, N_EVENT_BINS> labels = {
      "|z| < 10 cm and MB", "Good Calo-Cent", "Centrality 0-60%",
      "Leading Jet R=0.2", "Dijet R=0.2", "Leading Jet R=0.3", "Dijet R=0.3"};
  for (size_t i = 0; i < labels.size(); ++i)
  {
    m_hEvent->GetXaxis()->SetBinLabel(static_cast<int>(i) + 1, labels[i].c_str());
  }

  m_hCentrality = book<TH1D>("hCentrality", "Selected Events; Centrality [%]; Events", 60, -0.5, 59.5);

  for (size_t ir = 0; ir < m_n_radii; ++ir)
  {
    const char *r = m_radius_tag[ir];
    const std::string jet_label = std::format("R = 0.{}, |#eta| < {}, raw p_{{T}} #geq {} GeV", r[2], m_jet_eta_max[ir], m_jet_pt_raw_min);
    DijetHists &h = m_hists[ir];

    h.h2NjetsCent = book<TH2D>(std::format("h2NjetsCent_{}", r),
                               std::format("Jets ({}); Centrality [%]; N_{{jets}}", jet_label).c_str(),
                               60, -0.5, 59.5, 20, -0.5, 19.5);

    for (size_t ic = 0; ic < m_n_cent; ++ic)
    {
      const auto lo = static_cast<int>(m_cent_edges[ic]);
      const auto hi = static_cast<int>(m_cent_edges[ic + 1]);
      const std::string tag = std::format("{}_cent_{}_{}", r, lo, hi);
      const std::string cent_label = std::format("{}-{}%", lo, hi);

      for (size_t il = 0; il < m_n_lead; ++il)
      {
        const int plo = m_lead_edges[il];
        const int phi = m_lead_edges[il + 1];
        const std::string ltag = std::format("{}_ptlead_{}_{}", tag, plo, phi);
        const std::string label = std::format("Dijet ({}, {}, {} #leq raw p_{{T,lead}} < {} GeV)", jet_label, cent_label, plo, phi);

        h.h2PtSubDphi[ic][il] = book<TH2D>(std::format("h2PtSubDphi_{}", ltag),
                                           std::format("{}; Raw p_{{T,sub}} [GeV]; #Delta#phi", label).c_str(),
                                           static_cast<int>(m_sub_edges.size()) - 1, m_sub_edges.data(),
                                           m_bins_dphi, dphi_edges.data());
        h.h2XjDphi[ic][il] = book<TH2D>(std::format("h2XjDphi_{}", ltag),
                                        std::format("{}; x_{{J}} = p_{{T,sub}} / p_{{T,lead}}; #Delta#phi", label).c_str(),
                                        m_bins_xj, xj_edges.data(), m_bins_dphi, dphi_edges.data());
      }
    }

    // Timing and jet pT hists of the chosen dijet cut
    const double cut = m_pt_sub_min[ir];
    if (!(cut >= m_jet_pt_raw_min) || cut >= m_time_lead_upper.back())
    {
      throw std::runtime_error(std::format("pt_sub_min {} of {} must be in [{}, {}) GeV", cut, r, m_jet_pt_raw_min, m_time_lead_upper.back()));
    }
    h.time_lead_edges = {cut};
    for (double e : m_time_lead_upper)
    {
      if (e > cut)
      {
        h.time_lead_edges.push_back(e);
      }
    }
    const std::array<std::string, N_SEL> sel_label = {
        std::format("Dijet: #Delta#phi > 7#pi/8, raw p_{{T,sub}} #geq {} GeV", cut),
        std::format("Inclusive: raw p_{{T,lead}} #geq {} GeV", cut)};

    for (size_t is = 0; is < N_SEL; ++is)
    {
      const std::string stag = std::format("{}_{}", r, m_sel_time_tag[is]);
      for (size_t ic = 0; ic < m_n_cent; ++ic)
      {
        const auto clo = static_cast<int>(m_cent_edges[ic]);
        const auto chi = static_cast<int>(m_cent_edges[ic + 1]);
        auto &row = h.h2LeadTimeOHCalFrac[is][ic];
        row.clear();
        for (size_t ib = 0; ib + 1 < h.time_lead_edges.size(); ++ib)
        {
          const double plo = h.time_lead_edges[ib];
          const double phi = h.time_lead_edges[ib + 1];
          row.push_back(book<TH2D>(
              std::format("h2LeadTimeOHCalFrac_{}_cent_{}_{}_ptlead_{}_{}", stag, clo, chi, edge_tag(plo), edge_tag(phi)),
              std::format("Leading Jet, {} ({}, {}-{}%, {} #leq raw p_{{T,lead}} < {} GeV); OHCal Energy Fraction; Jet t - MBD t_{{0}} [ns]",
                          sel_label[is], jet_label, clo, chi, plo, phi).c_str(),
              m_bins_frac, m_frac_low, m_frac_high, m_bins_dt, -m_dt_max, m_dt_max));
        }
      }

      h.hLeadTimingStatus[is] = book<TH1D>(std::format("hLeadTimingStatus_{}", stag),
                                           std::format("Leading Jet Timing, {} ({}); ; Leading Jets", sel_label[is], jet_label).c_str(),
                                           5, 0, 5);
      const std::array<const char *, 5> slabels = {"Selected", "Filled", "t NaN", "OHCal frac or t0 NaN", "p_{T,lead} #geq 100 GeV"};
      for (size_t i = 0; i < slabels.size(); ++i)
      {
        h.hLeadTimingStatus[is]->GetXaxis()->SetBinLabel(static_cast<int>(i) + 1, slabels[i]);
      }

      const std::string jlabel = is == SEL_DIJET ? std::format("Jets in dijet events ({})", sel_label[is]) : std::string("Jets in all selected events");
      h.h2JetPtCent[is] = book<TH2D>(std::format("h2JetPtCent_{}_{}", r, m_sel_jet_tag[is]),
                                     std::format("{} ({}); Jet raw p_{{T}} [GeV]; Centrality [%]", jlabel, jet_label).c_str(),
                                     m_bins_jet_pt, 0, m_jet_pt_high, 60, -0.5, 59.5);
      // the same binning as the jet pT hists; every selected leading jet, also at >= 100 GeV and with t, t0 or x NaN
      h.h2LeadPtCent[is] = book<TH2D>(std::format("h2LeadPtCent_{}_{}", r, m_sel_time_tag[is]),
                                      std::format("Leading Jet, {} ({}); Leading jet raw p_{{T}} [GeV]; Centrality [%]", sel_label[is], jet_label).c_str(),
                                      m_bins_jet_pt, 0, m_jet_pt_high, 60, -0.5, 59.5);
    }
  }
}

// A bin edge as it appears in hist names: 14.25 -> "14p25", 20 -> "20"
std::string JetDijet::edge_tag(double x)
{
  std::string s = std::format("{:g}", x);
  std::replace(s.begin(), s.end(), '.', 'p');
  return s;
}

// The jet cuts: E > 0, |eta| < eta_max of the radius, raw pT >= m_jet_pt_raw_min
bool JetDijet::pass_jet(const JetData &jd, size_t idx, size_t ir) const
{
  return jd.e->at(idx) > 0 && std::abs(jd.eta->at(idx)) < m_jet_eta_max[ir] && jd.pt->at(idx) >= m_jet_pt_raw_min;
}

// Leading jet time - MBD t0 vs OHCal fraction in the cell (centrality class ic, pT_lead bin) of selection sel. As in
// Jet-Anav3, only jets with t, t0 and the OHCal fraction finite are filled; times outside +-25 ns go to the flow bins
void JetDijet::fill_lead_timing(size_t ir, Sel sel, size_t ic, size_t il)
{
  const JetData &jd = m_event_data.jets[ir];
  DijetHists &h = m_hists[ir];
  TH1 *status = h.hLeadTimingStatus[sel];
  status->Fill(0.5);

  const double pt_lead = jd.pt->at(il);
  const auto &edges = h.time_lead_edges;
  const auto it = std::upper_bound(edges.begin(), edges.end(), pt_lead);
  if (it == edges.begin() || it == edges.end())  // below pt_sub_min cannot happen; >= 100 GeV is outside the grid
  {
    status->Fill(4.5);
    return;
  }
  const auto ib = static_cast<size_t>(std::distance(edges.begin(), it) - 1);

  const double t = jd.t->at(il);
  if (!std::isfinite(t))
  {
    status->Fill(2.5);
    return;
  }
  const double frac = jd.ohcal_frac->at(il);
  if (!std::isfinite(frac) || !std::isfinite(m_event_data.t0))
  {
    status->Fill(3.5);
    return;
  }
  status->Fill(1.5);
  h.h2LeadTimeOHCalFrac[sel][ic][ib]->Fill(frac, t - m_event_data.t0);
}

int JetDijet::cent_class(double cent)
{
  for (size_t i = 0; i < m_n_cent; ++i)
  {
    if (cent >= m_cent_edges[i] && (cent < m_cent_edges[i + 1] || (i == m_n_cent - 1 && cent <= m_cent_edges[i + 1])))
    {
      return static_cast<int>(i);
    }
  }
  return -1;
}

// Leading and subleading jet (highest and second highest raw pT among jets with E > 0, |eta| < eta_max, raw pT >=
// m_jet_pt_raw_min)
void JetDijet::process_dijet(size_t ir, int cls)
{
  const JetData &jd = m_event_data.jets[ir];
  DijetHists &h = m_hists[ir];
  const double cent = m_event_data.centrality;

  int lead = -1;
  int sub = -1;
  int njets = 0;
  for (size_t idx = 0; idx < jd.pt->size(); ++idx)
  {
    if (!pass_jet(jd, idx, ir))
    {
      continue;
    }
    const double pt = jd.pt->at(idx);
    ++njets;
    h.h2JetPtCent[SEL_INCL]->Fill(pt, cent);  // every jet of every selected event
    const int i = static_cast<int>(idx);
    if (lead < 0 || pt > jd.pt->at(static_cast<size_t>(lead)))
    {
      sub = lead;
      lead = i;
    }
    else if (sub < 0 || pt > jd.pt->at(static_cast<size_t>(sub)))
    {
      sub = i;
    }
  }

  h.h2NjetsCent->Fill(cent, njets);
  if (lead < 0)
  {
    return;
  }

  const auto il = static_cast<size_t>(lead);
  const double pt_lead = jd.pt->at(il);
  m_hEvent->Fill(ir == 0 ? LEAD_R02 : LEAD_R03);

  const auto ic = static_cast<size_t>(cls);
  const double cut = m_pt_sub_min[ir];

  // Inclusive comparison: the leading jet above the cut, no subleading jet required
  if (pt_lead >= cut)
  {
    h.h2LeadPtCent[SEL_INCL]->Fill(pt_lead, cent);
    fill_lead_timing(ir, SEL_INCL, ic, il);
  }

  if (sub < 0)
  {
    return;
  }
  m_hEvent->Fill(ir == 0 ? DIJET_R02 : DIJET_R03);

  const auto is = static_cast<size_t>(sub);
  const double pt_sub = jd.pt->at(is);
  double dphi = std::abs(jd.phi->at(il) - jd.phi->at(is));
  if (dphi > std::numbers::pi)
  {
    dphi = 2 * std::numbers::pi - dphi;
  }
  // keep exactly back-to-back (and x_J = 1) in the last bin instead of the overflow
  dphi = std::min(dphi, std::nextafter(std::numbers::pi, 0.0));
  const double xj = std::min(pt_sub / pt_lead, std::nextafter(1.0, 0.0));

  // pT_lead class; leading jets at raw pT >= 100 GeV are counted only in hEvent
  const auto lead_it = std::upper_bound(m_lead_edges.begin(), m_lead_edges.end(), pt_lead);
  if (lead_it != m_lead_edges.begin() && lead_it != m_lead_edges.end())
  {
    const auto lc = static_cast<size_t>(std::distance(m_lead_edges.begin(), lead_it) - 1);
    h.h2PtSubDphi[ic][lc]->Fill(pt_sub, dphi);
    h.h2XjDphi[ic][lc]->Fill(xj, dphi);
  }

  // Dijet selection: back to back and the subleading jet above the cut (so the leading jet is too)
  if (dphi > m_dphi_b2b && pt_sub >= cut)
  {
    h.h2LeadPtCent[SEL_DIJET]->Fill(pt_lead, cent);
    fill_lead_timing(ir, SEL_DIJET, ic, il);
    for (size_t idx = 0; idx < jd.pt->size(); ++idx)  // every jet of the dijet event, third jets included
    {
      if (pass_jet(jd, idx, ir))
      {
        h.h2JetPtCent[SEL_DIJET]->Fill(jd.pt->at(idx), cent);
      }
    }
  }

  if (m_verbosity > 1)
  {
    std::cout << std::format("run {} event {} R {}: cent {:.0f}, lead {:.2f}, sub {:.2f}, dphi {:.3f}, x_J {:.3f}\n",
                             m_event_data.run, m_event_data.event, m_radius_tag[ir], cent, pt_lead, pt_sub, dphi, xj);
  }
}

void JetDijet::process_events()
{
  std::cout << "Processing... process_events" << std::endl;
  long long n_entries = m_chain->GetEntries();
  if (m_events_to_process > 0)
  {
    n_entries = std::min(m_events_to_process, n_entries);
  }

  for (long long event = 0; event < n_entries; ++event)
  {
    m_chain->GetEntry(event);

    if (event % 100000 == 0)
    {
      std::cout << std::format("Processing {}/{}: {:.2f} %", event, n_entries, static_cast<double>(event) * 100. / static_cast<double>(n_entries)) << std::endl;
    }

    // Event selection, as in Jet-Anav3
    m_hEvent->Fill(ENTRIES);
    if (!check_CaloMBD())
    {
      continue;
    }
    m_hEvent->Fill(CALOCENT);

    const double cent = m_event_data.centrality;
    const int cls = cent_class(cent);
    if (!(cent <= m_cent_high) || cls < 0)
    {
      continue;
    }
    m_hEvent->Fill(CENT);
    m_hCentrality->Fill(cent);

    for (size_t ir = 0; ir < m_n_radii; ++ir)
    {
      process_dijet(ir, cls);
    }
  }

  std::cout << "Finished... process_events" << std::endl;
}

void JetDijet::save_results() const
{
  std::filesystem::create_directories(m_output_dir);

  const std::string output_stem = std::filesystem::path(m_input_file).stem().string();
  const std::string output_filename = std::format("{}/Jet-Dijet_{}.root", m_output_dir, output_stem);

  auto output_file = std::make_unique<TFile>(output_filename.c_str(), "RECREATE");
  for (const auto &[name, hist] : m_hists_all)
  {
    hist->Write();
  }
  output_file->Close();

  std::cout << std::format("Events: {:.0f} entries, {:.0f} Good Calo-Cent, {:.0f} selected\n",
                           m_hEvent->GetBinContent(ENTRIES + 1), m_hEvent->GetBinContent(CALOCENT + 1), m_hEvent->GetBinContent(CENT + 1));
  std::cout << std::format("Results saved to: {}", output_filename) << std::endl;
}

// ====================================================================
// Main
// ====================================================================
int main(int argc, const char *const argv[])
{
  gROOT->SetBatch(true);
  TH1::AddDirectory(false);

  if (argc < 3 || argc > 9)
  {
    std::cout << "Usage: " << argv[0] << " input_file calo_mbd_file [events] [output_directory] [jet_pt_raw_min] [sigma_cut] [pt_sub_min_r02] [pt_sub_min_r03]" << std::endl;
    return 1;
  }

  int ctr = 1;
  const std::string input_file = argv[ctr++];
  const std::string calo_mbd_file = argv[ctr++];
  const long long events = (argc >= ctr + 1) ? std::atoll(argv[ctr++]) : 0;
  const std::string output_dir = (argc >= ctr + 1) ? argv[ctr++] : ".";
  const double jet_pt_raw_min = (argc >= ctr + 1) ? std::stod(argv[ctr++]) : 7;
  const double sigma_cut = (argc >= ctr + 1) ? std::stod(argv[ctr++]) : 3.5;
  const double pt_sub_min_r02 = (argc >= ctr + 1) ? std::stod(argv[ctr++]) : 14.25;
  const double pt_sub_min_r03 = (argc >= ctr + 1) ? std::stod(argv[ctr++]) : 21.5;

  std::cout << std::format("{:#<20}\n", "");
  std::cout << std::format("Run Params\n");
  std::cout << std::format("Input: {}\n", input_file);
  std::cout << std::format("Calo-MBD File: {}\n", calo_mbd_file);
  std::cout << std::format("Events: {}\n", events);
  std::cout << std::format("Output Dir: {}\n", output_dir);
  std::cout << std::format("Jet Raw pT Min: {} [GeV]\n", jet_pt_raw_min);
  std::cout << std::format("Sigma Cut: {:.2f}\n", sigma_cut);
  std::cout << std::format("Dijet pT_sub Min: R = 0.2: {} GeV, R = 0.3: {} GeV (raw; delta phi > 7pi/8)\n", pt_sub_min_r02, pt_sub_min_r03);
  std::cout << std::format("{:#<20}\n", "");

  try
  {
    JetDijet analysis(input_file, calo_mbd_file, events, output_dir);
    analysis.set_jet_pt_raw_min(jet_pt_raw_min);
    analysis.set_sigma_cut(sigma_cut);
    analysis.set_pt_sub_min(pt_sub_min_r02, pt_sub_min_r03);
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
