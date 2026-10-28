// Tell emacs that this is a C++ source
//  -*- C++ -*-.
#pragma once

#include "JetUtils.h"
#include "geometry_constants.h"

// -- sPHENIX
#include <fun4all/SubsysReco.h>

// -- c++
#include <array>
#include <cstdint>  // Required for std::uint8_t, std::uint16_t, etc.
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>
#include <numbers>

class PHCompositeNode;
class TFile;
class TTree;
class TH1;
class TH2;
class TProfile;
class TProfile2D;

class JetValidationv3 : public SubsysReco
{
 public:
  explicit JetValidationv3(const std::string &name = "JetValidationv3");

  int Init(PHCompositeNode *topNode) override;
  int InitRun(PHCompositeNode *topNode) override;
  int process_event(PHCompositeNode *topNode) override;
  // Clean up internals after each event.
  int ResetEvent(PHCompositeNode *topNode) override;
  int End(PHCompositeNode *topNode) override;

  void set_do_unsub(bool b = true) { m_do_unsub = b; }
  void set_do_iter(bool b = true) { m_do_iter = b; }
  void set_do_iter_noflow(bool b = true) { m_do_iter_noflow = b; }
  void set_do_mult(bool b = true) { m_do_mult = b; }
  void set_do_mult_noflow(bool b = true) { m_do_mult_noflow = b; }
  void set_do_detailed(bool b = true) { m_do_detailed = b; }
  // store EMfrac-calibrated pT (nodes "<calib node>_calib_emfrac") and per-jet calo fractions
  void set_do_emfrac(bool b = true) { m_do_emfrac = b; }
  // store per-jet timing (jet time and per calo layer time) and per-jet calo fractions
  void set_do_timing(bool b = true) { m_do_timing = b; }
  void set_timing_e_threshold(double e) { m_timing_e_threshold = e; }
  void set_do_r02(bool b = true) { m_do_r02 = b; }
  void set_do_r03(bool b = true) { m_do_r03 = b; }
  void set_jet_pt_min(double pt_min) { m_jet_pt_min_cut = pt_min; }

  void set_mult_r02_node_names(const std::string& raw, const std::string& calib) { m_recoJetName_mult_r02 = raw; m_recoJetName_mult_calib_r02 = calib; }
  void set_mult_r03_node_names(const std::string& raw, const std::string& calib) { m_recoJetName_mult_r03 = raw; m_recoJetName_mult_calib_r03 = calib; }
  void set_mult_tower_node_names(const std::string& cemc, const std::string& ihcal, const std::string& ohcal) { m_towerNode_mult_cemc = cemc; m_towerNode_mult_ihcal = ihcal; m_towerNode_mult_ohcal = ohcal; }

  void set_jet_radii(const std::vector<float>& radii);
  void set_jet_radii(const std::vector<double>& radii);
  void set_jet_radii(std::initializer_list<double> radii);

  bool get_do_unsub() const { return m_do_unsub; }
  bool get_do_iter() const { return m_do_iter; }
  bool get_do_iter_noflow() const { return m_do_iter_noflow; }
  bool get_do_mult() const { return m_do_mult; }
  bool get_do_mult_noflow() const { return m_do_mult_noflow; }
  bool get_do_detailed() const { return m_do_detailed; }
  bool get_do_emfrac() const { return m_do_emfrac; }
  bool get_do_timing() const { return m_do_timing; }
  double get_timing_e_threshold() const { return m_timing_e_threshold; }
  bool get_do_r02() const { return m_do_r02; }
  bool get_do_r03() const { return m_do_r03; }
  double get_jet_pt_min() const { return m_jet_pt_min_cut; }
  std::vector<float> get_jet_radii() const;

 private:
  int process_UE(PHCompositeNode *topNode);
  int process_jets(PHCompositeNode *topNode);

  struct JetData
  {
    std::vector<double> pt;        // NOLINT(misc-non-private-member-variables-in-classes)
    std::vector<double> pt_calib;  // NOLINT(misc-non-private-member-variables-in-classes)
    std::vector<double> e;         // NOLINT(misc-non-private-member-variables-in-classes)
    std::vector<double> phi;       // NOLINT(misc-non-private-member-variables-in-classes)
    std::vector<double> eta;       // NOLINT(misc-non-private-member-variables-in-classes)
    double max_pt{0};              // NOLINT(misc-non-private-member-variables-in-classes)

    // EMfrac JES comparison (filled only with m_do_emfrac)
    std::vector<double> pt_calib_emfrac;  // NOLINT(misc-non-private-member-variables-in-classes)

    // calo fractions (filled with m_do_emfrac or m_do_timing)
    std::vector<double> emcal_frac;       // NOLINT(misc-non-private-member-variables-in-classes)
    std::vector<double> ihcal_frac;       // NOLINT(misc-non-private-member-variables-in-classes)
    std::vector<double> ohcal_frac;       // NOLINT(misc-non-private-member-variables-in-classes)

    // timing [ns] (filled only with m_do_timing): jet time and energy-weighted time per calo layer
    std::vector<double> t;        // NOLINT(misc-non-private-member-variables-in-classes)
    std::vector<double> t_emcal;  // NOLINT(misc-non-private-member-variables-in-classes)
    std::vector<double> t_ihcal;  // NOLINT(misc-non-private-member-variables-in-classes)
    std::vector<double> t_ohcal;  // NOLINT(misc-non-private-member-variables-in-classes)

    // Detailed tower constituent info (nested vectors: 1 vector per jet in the event)
    std::vector<std::vector<int>> emcal_tower_index;        // NOLINT(misc-non-private-member-variables-in-classes)
    std::vector<std::vector<double>> emcal_tower_energy;   // NOLINT(misc-non-private-member-variables-in-classes)
    std::vector<std::vector<double>> emcal_tower_pt;       // NOLINT(misc-non-private-member-variables-in-classes)

    std::vector<std::vector<int>> ihcal_tower_index;        // NOLINT(misc-non-private-member-variables-in-classes)
    std::vector<std::vector<double>> ihcal_tower_energy;   // NOLINT(misc-non-private-member-variables-in-classes)
    std::vector<std::vector<double>> ihcal_tower_pt;       // NOLINT(misc-non-private-member-variables-in-classes)

    std::vector<std::vector<int>> ohcal_tower_index;        // NOLINT(misc-non-private-member-variables-in-classes)
    std::vector<std::vector<double>> ohcal_tower_energy;   // NOLINT(misc-non-private-member-variables-in-classes)
    std::vector<std::vector<double>> ohcal_tower_pt;       // NOLINT(misc-non-private-member-variables-in-classes)

    void clear()
    {
      pt.clear();
      pt_calib.clear();
      e.clear();
      phi.clear();
      eta.clear();
      max_pt = 0;

      pt_calib_emfrac.clear();
      emcal_frac.clear();
      ihcal_frac.clear();
      ohcal_frac.clear();

      t.clear();
      t_emcal.clear();
      t_ihcal.clear();
      t_ohcal.clear();

      emcal_tower_index.clear();
      emcal_tower_energy.clear();
      emcal_tower_pt.clear();

      ihcal_tower_index.clear();
      ihcal_tower_energy.clear();
      ihcal_tower_pt.clear();

      ohcal_tower_index.clear();
      ohcal_tower_energy.clear();
      ohcal_tower_pt.clear();
    }
  };

  struct EventData
  {
    // UE
    // key: iter   -> UE Iterative Method
    //      iternf -> UE Iterative Method w/o flow modulation
    //      mult   -> UE Multiplicity Method (New)
    //      multnf -> UE Multiplicity Method w/o flow modulation
    float calo_v2_iter{0};
    float calo_v2_mult{0};

    int seeds_iter{0};
    int seeds_iternf{0};
    int seeds_mult{0};
    int seeds_multnf{0};

    bool is_flow_failure_iter{false};
    bool is_flow_failure_mult{false};

    // jets
    JetData iter_r02;
    JetData iter_r03;
    JetData iternf_r02;
    JetData iternf_r03;
    JetData mult_r02;
    JetData mult_r03;
    JetData multnf_r02;
    JetData multnf_r03;
    JetData unsub_r02;
    JetData unsub_r03;
  };

  EventData m_data;

  std::string m_recoJetName_iter_r02{"AntiKt_Tower_r02_Sub1"};
  std::string m_recoJetName_mult_r02{"AntiKt_Tower_r02_MultSub1"};
  std::string m_recoJetName_iter_calib_r02{"AntiKt_Tower_r02_Sub1_calib"};
  std::string m_recoJetName_mult_calib_r02{"AntiKt_Tower_r02_MultSub1_calib"};

  std::string m_recoJetName_iter_r03{"AntiKt_Tower_r03_Sub1"};
  std::string m_recoJetName_mult_r03{"AntiKt_Tower_r03_MultSub1"};
  std::string m_recoJetName_iter_calib_r03{"AntiKt_Tower_r03_Sub1_calib"};
  std::string m_recoJetName_mult_calib_r03{"AntiKt_Tower_r03_MultSub1_calib"};

  std::string m_recoJetName_iternf_r02{"AntiKt_Tower_r02_NoFlowSub1"};
  std::string m_recoJetName_iternf_calib_r02{"AntiKt_Tower_r02_NoFlowSub1_calib"};
  std::string m_recoJetName_iternf_r03{"AntiKt_Tower_r03_NoFlowSub1"};
  std::string m_recoJetName_iternf_calib_r03{"AntiKt_Tower_r03_NoFlowSub1_calib"};

  std::string m_towerNode_iternf_cemc{"NOFLOW_TOWERINFO_CALIB_CEMC_RETOWER_SUB1"};
  std::string m_towerNode_iternf_ihcal{"NOFLOW_TOWERINFO_CALIB_HCALIN_SUB1"};
  std::string m_towerNode_iternf_ohcal{"NOFLOW_TOWERINFO_CALIB_HCALOUT_SUB1"};

  std::string m_towerNode_mult_cemc{"MULTSUB_TOWERINFO_CALIB_CEMC_RETOWER_SUB1"};
  std::string m_towerNode_mult_ihcal{"MULTSUB_TOWERINFO_CALIB_HCALIN_SUB1"};
  std::string m_towerNode_mult_ohcal{"MULTSUB_TOWERINFO_CALIB_HCALOUT_SUB1"};

  std::string m_recoJetName_multnf_r02{"AntiKt_Tower_r02_MultNoFlowSub1"};
  std::string m_recoJetName_multnf_calib_r02{"AntiKt_Tower_r02_MultNoFlowSub1_calib"};
  std::string m_recoJetName_multnf_r03{"AntiKt_Tower_r03_MultNoFlowSub1"};
  std::string m_recoJetName_multnf_calib_r03{"AntiKt_Tower_r03_MultNoFlowSub1_calib"};

  std::string m_towerNode_multnf_cemc{"MULTSUB_NOFLOW_TOWERINFO_CALIB_CEMC_RETOWER_SUB1"};
  std::string m_towerNode_multnf_ihcal{"MULTSUB_NOFLOW_TOWERINFO_CALIB_HCALIN_SUB1"};
  std::string m_towerNode_multnf_ohcal{"MULTSUB_NOFLOW_TOWERINFO_CALIB_HCALOUT_SUB1"};

  std::string m_recoJetName_unsub_r02{"AntiKt_TowerInfo_r02"};
  std::string m_recoJetName_unsub_calib_r02{"AntiKt_TowerInfo_r02_calib"};
  std::string m_recoJetName_unsub_r03{"AntiKt_TowerInfo_r03"};
  std::string m_recoJetName_unsub_calib_r03{"AntiKt_TowerInfo_r03_calib"};

  double m_jet_pt_min_cut{10};     // GeV

  // towers enter the jet time only above this energy, same as TowerJetInput (GeV)
  double m_timing_e_threshold{0.1};
  // tower time is in samples
  static constexpr double NS_PER_SAMPLE = 17.6;

  // EMfrac-calibrated jet node = legacy calib node name + this suffix
  std::string m_emfrac_calib_suffix{"_emfrac"};

  bool m_do_unsub{true};
  bool m_do_iter{true};
  bool m_do_iter_noflow{false};
  bool m_do_mult{true};
  bool m_do_mult_noflow{false};
  bool m_do_detailed{false};
  bool m_do_emfrac{false};
  bool m_do_timing{false};
  bool m_do_r02{true};
  bool m_do_r03{true};

  static constexpr size_t N_HCAL_TOWERS = CaloGeometry::HCAL_ETA_BINS * CaloGeometry::HCAL_PHI_BINS;

  struct TowerGeomInfo
  {
    double eta = 0.0;
    double phi = 0.0;
    double z0 = 0.0;
    bool is_valid = false;
  };

  using TowerGeomArray = std::array<TowerGeomInfo, N_HCAL_TOWERS>;

  double m_r_cemc = JetUtils::radius_EM;
  double m_r_hcalin = JetUtils::radius_IH;
  double m_r_hcalout = JetUtils::radius_OH;

  TowerGeomArray m_geom_cemc{};
  TowerGeomArray m_geom_hcalin{};
  TowerGeomArray m_geom_hcalout{};
};
