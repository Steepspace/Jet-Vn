// Tell emacs that this is a C++ source
//  -*- C++ -*-.
#pragma once

#include <fun4all/SubsysReco.h>

#include <array>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <numbers>
#include <string>
#include <vector>

class PHCompositeNode;
class Eventplaneinfo;
class TH1;
class TH2;
class TProfile;
class TProfile2D;

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

  // Event plane calibration status of the event, as applied by EventPlaneReco.
  // EventPlaneReco stores the raw Q vector in place of the calibrated one (and a zero
  // recentered Q vector) when it skips the calibration.
  enum EPStatus : int
  {
    EP_CALIBRATED = 0,
    EP_ZERO_CHARGE = 1,     // zero sEPD charge in an arm: raw Q = (0, 0)
    EP_CENT_OUT_RANGE = 2,  // centrality invalid or >= m_ep_calib_cent_max: raw Q used
    EP_NO_CALIB = 3,        // no calibration loaded for the run: raw Q used
    EP_NON_FINITE = 4,      // NaN / inf Q vector (e.g. a non-finite sEPD channel charge in EventPlaneReco)
    EP_NSTATUS = 5
  };
  EPStatus get_ep_status(const Eventplaneinfo *epd_S, const Eventplaneinfo *epd_N, const Eventplaneinfo *epd_NS) const;

  bool has_valid_cent() const;

  // Event plane detectors (South, North, North+South) and Q vector calibration stages
  static constexpr unsigned int kNEPDet{3};
  static constexpr std::array<unsigned int, 2> kEPHarmonics{2, 3};  // Q vector harmonics stored in the tree
  static constexpr unsigned int kNQStage{3};      // raw, recentered, recentered + flattened
  static constexpr unsigned int kNQMoment{5};     // Qx, Qy, Qx^2, Qy^2, QxQy
  static constexpr unsigned int kNFlatTerms{4};   // cos/sin(2k Psi_2), k = 1..4
  static constexpr unsigned int kNArm{2};         // sEPD South, North
  static constexpr std::array<const char *, kNArm> kArmNames{"S", "N"};  // tree branch suffixes

  // EventPlaneReco calibrates centralities in [0, m_ep_calib_cent_max) %
  double m_ep_calib_cent_max{80};

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
    // phi bins: 24 for rings 1-15, 12 for ring 0
    unsigned int bins_sepd_phi{24};
    double sepd_phi_low{-0.5};
    double sepd_phi_high{23.5};

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
    TH1 *hSEPD_Channel_NonFinite{nullptr};  // channels skipped for NaN / inf charge
    TProfile *pSEPD_Ring_Charge_South{nullptr};
    TProfile *pSEPD_Ring_Charge_North{nullptr};

    // sEPD (ring, phi) maps per arm [0 = South, 1 = North]: hit counts (charge above threshold)
    // and mean charge of the hits. Occupancy = hits / hSEPD_NEvents.
    TH1 *hSEPD_NEvents{nullptr};
    std::array<TH2 *, kNArm> h2SEPD_RingPhi_Hits{};
    std::array<TProfile2D *, kNArm> pSEPD_RingPhi_Charge{};

    // sEPD total charge vs centrality
    TH2 *h2SEPD_Charge_Centrality{nullptr};

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

    // Event Plane - calibration status (EPStatus)
    TH1 *hEP_Status{nullptr};

    // Event Plane - Q_2 moments vs centrality [stage][det][moment], calibrated events
    // (recentering: <Qx> = <Qy> = 0; flattening: <Qx^2> = <Qy^2>, <QxQy> = 0)
    std::array<std::array<std::array<TProfile *, kNQMoment>, kNEPDet>, kNQStage> pQ2_Moment{};

    // Event Plane - Fourier flatness vs centrality [det][k-1]: <cos(2k Psi_2)>, <sin(2k Psi_2)>, calibrated events
    std::array<std::array<TProfile *, kNFlatTerms>, kNEPDet> pFlat_Cos{};
    std::array<std::array<TProfile *, kNFlatTerms>, kNEPDet> pFlat_Sin{};

    // Event Plane - extra sub-event checks: <sin(2 dPsi_2)> = 0, n = 3 resolution
    TProfile *pSin2DeltaPsi2_Cent{nullptr};
    TProfile *pCos3DeltaPsi3_Cent{nullptr};
  };

  AnalysisHists m_hists;

  struct EventData
  {
    // sEPD - Psi_2 of the calibrated North+South Q vector, as used by the UE subtraction
    // (DetermineTowerBackground, do_flow = 3: get_shifted_psi(2)); NaN if not filled or Q = (0, 0)
    double psi2_NS{std::numeric_limits<double>::quiet_NaN()};

    // sEPD - Q vectors per arm (South, North), each indexed by harmonic (kEPHarmonics: n = 2, 3).
    // Normalized by the arm's sEPD charge (EventPlaneReco); Psi_n = atan2(qy, qx) / n.
    // qx/qy: recentered + flattened, equal to the raw Q unless ep_status == EP_CALIBRATED
    std::array<std::vector<float>, kNArm> qx_raw;
    std::array<std::vector<float>, kNArm> qy_raw;
    std::array<std::vector<float>, kNArm> qx;
    std::array<std::vector<float>, kNArm> qy;
    int ep_status{-1};  // EPStatus, -1 if not filled

    // sEPD - QA
    double sepd_charge_south{0};
    double sepd_charge_north{0};

    // MBD - QA
    double mbd_charge_south{0};
    double mbd_charge_north{0};

    // Centrality [%] (mbd_NS), NaN if CentralityInfo is missing
    double cent{std::numeric_limits<double>::quiet_NaN()};
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
