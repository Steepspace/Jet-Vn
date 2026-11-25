// Tell emacs that this is a C++ source
//  -*- C++ -*-.
#pragma once

#include <fun4all/SubsysReco.h>

#include <cstdint>
#include <map>
#include <memory>
#include <numbers>
#include <string>
#include <vector>

class PHCompositeNode;
class TH1;
class TH2;
class TProfile;

class GlobalQA : public SubsysReco
{
 public:
  explicit GlobalQA(const std::string &name = "GlobalQA");
  ~GlobalQA() override = default;

  // -- Rule of Five: explicitly delete copy and move semantics --
  GlobalQA(const GlobalQA &) = delete;
  GlobalQA &operator=(const GlobalQA &) = delete;
  GlobalQA(GlobalQA &&) = delete;
  GlobalQA &operator=(GlobalQA &&) = delete;

  int Init(PHCompositeNode *topNode) override;
  int process_event(PHCompositeNode *topNode) override;
  int ResetEvent(PHCompositeNode *topNode) override;
  int End(PHCompositeNode *topNode) override;

  void set_do_ep(bool b = true) { m_do_ep = b; }
  void set_do_sepd(bool b = true) { m_do_sepd = b; }
  void set_do_mbd(bool b = true) { m_do_mbd = b; }
  void set_do_hist(bool b = true) { m_do_hist = b; }
  void set_do_tree(bool b = true) { m_do_tree = b; }

  bool get_do_ep() const { return m_do_ep; }
  bool get_do_sepd() const { return m_do_sepd; }
  bool get_do_mbd() const { return m_do_mbd; }
  bool get_do_hist() const { return m_do_hist; }
  bool get_do_tree() const { return m_do_tree; }

  void set_sepd_threshold(double threshold) { m_sepd_channel_threshold = threshold; }
  void set_skipRing0(bool b = true) { m_skipRing0 = b; }

  double get_sepd_charge_south() const { return m_data.sepd_charge_south; }
  double get_sepd_charge_north() const { return m_data.sepd_charge_north; }
  double get_sepd_total_charge() const { return m_data.sepd_charge_south + m_data.sepd_charge_north; }

  double get_mbd_charge_south() const { return m_data.mbd_charge_south; }
  double get_mbd_charge_north() const { return m_data.mbd_charge_north; }
  double get_mbd_total_charge() const { return m_data.mbd_charge_south + m_data.mbd_charge_north; }

 private:
  int process_event_plane(PHCompositeNode *topNode);
  int process_sepd(PHCompositeNode *topNode);
  int process_mbd(PHCompositeNode *topNode);

  struct HistConfig
  {
    // sEPD charge
    unsigned int bins_sepd_charge{250};
    double sepd_charge_arm_low{0};
    double sepd_charge_arm_high{2.5e4};
    double sepd_charge_total_low{0};
    double sepd_charge_total_high{2.5e4};

    // sEPD hits (at most 744 channels total, 372 channels per arm)
    unsigned int bins_sepd_hits_arm{373};
    double sepd_hits_arm_low{-0.5};
    double sepd_hits_arm_high{372.5};
    unsigned int bins_sepd_hits_total{745};
    double sepd_hits_total_low{-0.5};
    double sepd_hits_total_high{744.5};

    // sEPD channel & ring
    unsigned int bins_sepd_channels{744};
    double sepd_channel_low{-0.5};
    double sepd_channel_high{743.5};
    unsigned int bins_sepd_channel_charge{200};
    double sepd_channel_charge_low{0};
    double sepd_channel_charge_high{200};
    unsigned int bins_sepd_rings{16};
    double sepd_ring_low{-0.5};
    double sepd_ring_high{15.5};

    // MBD charge (total charge > 2100 excluded by MB classifier)
    unsigned int bins_mbd_charge{210};
    double mbd_charge_arm_low{0};
    double mbd_charge_arm_high{2100.0};
    double mbd_charge_total_low{0};
    double mbd_charge_total_high{2100.0};

    // Event plane (2*Psi_2 in [-pi, pi])
    unsigned int bins_psi{126};
    double psi_low{-std::numbers::pi};
    double psi_high{std::numbers::pi};

    // Centrality (centered on integers: bin center = integer centile)
    unsigned int bins_cent{100};
    double cent_low{-0.5};
    double cent_high{99.5};
    unsigned int bins_cent_profile{20};
  };

  HistConfig m_hist_config;

  struct AnalysisHists
  {
    // sEPD QA (Individual South/North distributions available via 2D projections)
    TH1 *hSEPD_Charge_Total{nullptr};
    TH1 *hSEPD_Hits_Total{nullptr};
    TH2 *h2SEPD_Charge_South_North{nullptr};
    TH2 *h2SEPD_Hits_South_North{nullptr};
    TH2 *h2SEPD_Channel_Charge{nullptr};
    TProfile *pSEPD_Channel_Charge{nullptr};
    TProfile *pSEPD_Ring_Charge_South{nullptr};
    TProfile *pSEPD_Ring_Charge_North{nullptr};

    // MBD QA (Individual South/North distributions available via 2D projection)
    TH1 *hMBD_Charge_Total{nullptr};
    TH2 *h2MBD_Charge_South_North{nullptr};

    // Cross-detector (sEPD vs MBD)
    TH2 *h2SEPD_MBD_Total_Charge{nullptr};

    // Event Plane - 2D & Correlations (1D angles available via 2D projections)
    TH2 *h2Psi2_raw_SN{nullptr};
    TH2 *h2Psi2_SN{nullptr};
    TH1 *hDeltaPsi2_SN{nullptr};
    TH1 *hCos2DeltaPsi2{nullptr};

    // Event Plane - vs Centrality (if CentralityInfo available)
    TH2 *h2Psi2_S_raw{nullptr};
    TH2 *h2Psi2_N_raw{nullptr};
    TH2 *h2Psi2_NS_raw{nullptr};
    TH2 *h2Psi2_S{nullptr};
    TH2 *h2Psi2_N{nullptr};
    TH2 *h2Psi2_NS{nullptr};
    TProfile *pCos2DeltaPsi2_Cent{nullptr};
  };

  AnalysisHists m_hists;

  struct EventData
  {
    // sEPD - Event Plane
    double psi2_raw_S{0};
    double psi2_raw_N{0};
    double psi2_raw_NS{0};

    double psi2_S{0};
    double psi2_N{0};
    double psi2_NS{0};

    // sEPD - QA
    double sepd_charge_south{0};
    double sepd_charge_north{0};

    // MBD - QA
    double mbd_charge_south{0};
    double mbd_charge_north{0};
  };

  EventData m_data;

  double m_sepd_channel_threshold{0.5};

  bool m_skipRing0{true};

  bool m_do_ep{true};
  bool m_do_sepd{true};
  bool m_do_mbd{true};
  bool m_do_hist{true};
  bool m_do_tree{true};
};
