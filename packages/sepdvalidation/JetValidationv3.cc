#include "JetValidationv3.h"

// -- c++
#include <format>
#include <iostream>
#include <limits>

// -- event
#include <ffaobjects/EventHeader.h>

// -- Fun4All
#include <fun4all/Fun4AllReturnCodes.h>
#include <fun4all/Fun4AllServer.h>

// -- Nodes
#include <phool/PHCompositeNode.h>
#include <phool/getClass.h>

// -- calo
#include <calobase/RawTowerDefs.h>
#include <calobase/RawTowerGeom.h>
#include <calobase/RawTowerGeomContainer.h>
#include <calobase/TowerInfo.h>
#include <calobase/TowerInfoContainer.h>

// -- jet
#include <jetbase/Jet.h>
#include <jetbase/JetContainer.h>

// -- jetbackground
#include <jetbackground/TowerBackground.h>

#include <treefiller/TreeFiller.h>

#include "JetUtils.h"

// -- vertex
#include <globalvertex/GlobalVertex.h>
#include <globalvertex/GlobalVertexMap.h>

// -- ROOT
#include <TH1.h>
#include <TH2.h>
#include <TProfile2D.h>
#include <TTree.h>

namespace
{
  // calo layer of a jet constituent: 0 = EMCal, 1 = IHCal, 2 = OHCal, -1 = not a calo tower
  int get_calo_layer(Jet::SRC src)
  {
    switch (src)
    {
    case Jet::SRC::CEMC_TOWERINFO_RETOWER:
    case Jet::SRC::CEMC_TOWERINFO_SUB1:
      return 0;
    case Jet::SRC::HCALIN_TOWERINFO:
    case Jet::SRC::HCALIN_TOWERINFO_SUB1:
      return 1;
    case Jet::SRC::HCALOUT_TOWERINFO:
    case Jet::SRC::HCALOUT_TOWERINFO_SUB1:
      return 2;
    default:
      return -1;
    }
  }
}  // namespace

//____________________________________________________________________________..
JetValidationv3::JetValidationv3(const std::string &name)
  : SubsysReco(name)
{
}

void JetValidationv3::set_jet_radii(const std::vector<float>& radii)
{
  m_do_r02 = false;
  m_do_r03 = false;
  for (float r : radii)
  {
    if (std::abs(r - 0.2f) < 1e-3f) m_do_r02 = true;
    if (std::abs(r - 0.3f) < 1e-3f) m_do_r03 = true;
  }
}

void JetValidationv3::set_jet_radii(const std::vector<double>& radii)
{
  m_do_r02 = false;
  m_do_r03 = false;
  for (double r : radii)
  {
    if (std::abs(r - 0.2) < 1e-3) m_do_r02 = true;
    if (std::abs(r - 0.3) < 1e-3) m_do_r03 = true;
  }
}

void JetValidationv3::set_jet_radii(std::initializer_list<double> radii)
{
  set_jet_radii(std::vector<double>(radii));
}

std::vector<float> JetValidationv3::get_jet_radii() const
{
  std::vector<float> r;
  if (m_do_r02)
  {
    r.push_back(0.2F);
  }
  if (m_do_r03)
  {
    r.push_back(0.3F);
  }
  return r;
}

//____________________________________________________________________________..
int JetValidationv3::Init([[maybe_unused]] PHCompositeNode *topNode)
{
  Fun4AllServer *se = Fun4AllServer::instance();
  se->Print("NODETREE");

  TTree* tree = TreeFiller::getTree();
  if (!tree)
  {
    std::cout << "JetValidationv3: Failed to get/create TTree" << std::endl;
    return Fun4AllReturnCodes::ABORTRUN;
  }

  // books the per-jet branches "<field>_<tag>", e.g. pt_iter_r02
  auto book_jets = [&](JetData& jd, const std::string& tag)
  {
    tree->Branch(("max_pt_" + tag).c_str(), &jd.max_pt);
    tree->Branch(("pt_" + tag).c_str(), &jd.pt);
    tree->Branch(("pt_calib_" + tag).c_str(), &jd.pt_calib);
    tree->Branch(("e_" + tag).c_str(), &jd.e);
    tree->Branch(("phi_" + tag).c_str(), &jd.phi);
    tree->Branch(("eta_" + tag).c_str(), &jd.eta);

    if (m_do_emfrac)
    {
      tree->Branch(("pt_calib_emfrac_" + tag).c_str(), &jd.pt_calib_emfrac);
    }

    if (m_do_emfrac || m_do_timing)
    {
      tree->Branch(("emcal_frac_" + tag).c_str(), &jd.emcal_frac);
      tree->Branch(("ihcal_frac_" + tag).c_str(), &jd.ihcal_frac);
      tree->Branch(("ohcal_frac_" + tag).c_str(), &jd.ohcal_frac);
    }

    if (m_do_timing)
    {
      tree->Branch(("t_" + tag).c_str(), &jd.t);
      tree->Branch(("t_emcal_" + tag).c_str(), &jd.t_emcal);
      tree->Branch(("t_ihcal_" + tag).c_str(), &jd.t_ihcal);
      tree->Branch(("t_ohcal_" + tag).c_str(), &jd.t_ohcal);
    }

    if (m_do_lead_const)
    {
      tree->Branch(("lead_pt_" + tag).c_str(), &jd.lead_pt);
      tree->Branch(("lead_e_" + tag).c_str(), &jd.lead_e);
      tree->Branch(("lead_calo_" + tag).c_str(), &jd.lead_calo);
      tree->Branch(("lead_index_" + tag).c_str(), &jd.lead_index);
    }

    if (m_do_detailed)
    {
      tree->Branch(("emcal_tower_index_" + tag).c_str(), &jd.emcal_tower_index);
      tree->Branch(("emcal_tower_energy_" + tag).c_str(), &jd.emcal_tower_energy);
      tree->Branch(("emcal_tower_pt_" + tag).c_str(), &jd.emcal_tower_pt);

      tree->Branch(("ihcal_tower_index_" + tag).c_str(), &jd.ihcal_tower_index);
      tree->Branch(("ihcal_tower_energy_" + tag).c_str(), &jd.ihcal_tower_energy);
      tree->Branch(("ihcal_tower_pt_" + tag).c_str(), &jd.ihcal_tower_pt);

      tree->Branch(("ohcal_tower_index_" + tag).c_str(), &jd.ohcal_tower_index);
      tree->Branch(("ohcal_tower_energy_" + tag).c_str(), &jd.ohcal_tower_energy);
      tree->Branch(("ohcal_tower_pt_" + tag).c_str(), &jd.ohcal_tower_pt);
    }
  };

  auto book_radii = [&](JetData& jd_r02, JetData& jd_r03, const std::string& method)
  {
    if (m_do_r02)
    {
      book_jets(jd_r02, method + "_r02");
    }
    if (m_do_r03)
    {
      book_jets(jd_r03, method + "_r03");
    }
  };

  if (m_do_iter)
  {
    tree->Branch("seeds_iter", &m_data.seeds_iter);
    tree->Branch("calo_v2_iter", &m_data.calo_v2_iter);
    tree->Branch("is_flow_failure_iter", &m_data.is_flow_failure_iter);
    book_radii(m_data.iter_r02, m_data.iter_r03, "iter");
  }

  if (m_do_iter_noflow)
  {
    tree->Branch("seeds_iternf", &m_data.seeds_iternf);
    book_radii(m_data.iternf_r02, m_data.iternf_r03, "iternf");
  }

  if (m_do_mult)
  {
    tree->Branch("seeds_mult", &m_data.seeds_mult);
    tree->Branch("calo_v2_mult", &m_data.calo_v2_mult);
    tree->Branch("is_flow_failure_mult", &m_data.is_flow_failure_mult);
    book_radii(m_data.mult_r02, m_data.mult_r03, "mult");
  }

  if (m_do_mult_noflow)
  {
    tree->Branch("seeds_multnf", &m_data.seeds_multnf);
    book_radii(m_data.multnf_r02, m_data.multnf_r03, "multnf");
  }

  if (m_do_unsub)
  {
    book_radii(m_data.unsub_r02, m_data.unsub_r03, "unsub");
  }

  return Fun4AllReturnCodes::EVENT_OK;
}

//____________________________________________________________________________..
int JetValidationv3::InitRun(PHCompositeNode *topNode)
{
  m_geom_cemc.fill(TowerGeomInfo{});
  m_geom_hcalin.fill(TowerGeomInfo{});
  m_geom_hcalout.fill(TowerGeomInfo{});

  RawTowerGeomContainer* geom_hcalin = findNode::getClass<RawTowerGeomContainer>(topNode, "TOWERGEOM_HCALIN");
  RawTowerGeomContainer* geom_hcalout = findNode::getClass<RawTowerGeomContainer>(topNode, "TOWERGEOM_HCALOUT");

  if (geom_hcalin)
  {
    for (unsigned int ieta = 0; ieta < CaloGeometry::HCAL_ETA_BINS; ++ieta)
    {
      for (unsigned int iphi = 0; iphi < CaloGeometry::HCAL_PHI_BINS; ++iphi)
      {
        RawTowerDefs::keytype key = RawTowerDefs::encode_towerid(RawTowerDefs::CalorimeterId::HCALIN, ieta, iphi);
        RawTowerGeom *geom = geom_hcalin->get_tower_geometry(key);
        if (geom)
        {
          double eta = geom->get_eta();
          double phi = geom->get_phi();
          double sinh_eta = std::sinh(eta);
          size_t idx = static_cast<size_t>(ieta) * CaloGeometry::HCAL_PHI_BINS + iphi;

          m_geom_hcalin[idx] = {eta, phi, sinh_eta * m_r_hcalin, true};
          m_geom_cemc[idx] = {eta, phi, sinh_eta * m_r_cemc, true};
        }
      }
    }
  }

  if (geom_hcalout)
  {
    for (unsigned int ieta = 0; ieta < CaloGeometry::HCAL_ETA_BINS; ++ieta)
    {
      for (unsigned int iphi = 0; iphi < CaloGeometry::HCAL_PHI_BINS; ++iphi)
      {
        RawTowerDefs::keytype key = RawTowerDefs::encode_towerid(RawTowerDefs::CalorimeterId::HCALOUT, ieta, iphi);
        RawTowerGeom *geom = geom_hcalout->get_tower_geometry(key);
        if (geom)
        {
          double eta = geom->get_eta();
          double phi = geom->get_phi();
          double sinh_eta = std::sinh(eta);
          size_t idx = static_cast<size_t>(ieta) * CaloGeometry::HCAL_PHI_BINS + iphi;

          m_geom_hcalout[idx] = {eta, phi, sinh_eta * m_r_hcalout, true};
        }
      }
    }
  }

  return Fun4AllReturnCodes::EVENT_OK;
}

//____________________________________________________________________________..
int JetValidationv3::process_UE(PHCompositeNode *topNode)
{
  EventHeader *eventInfo = findNode::getClass<EventHeader>(topNode, "EventHeader");
  int event_id = eventInfo ? eventInfo->get_EvtSequence() : -1;

  if (m_do_iter)
  {
    TowerBackground* towerBkg_iter = findNode::getClass<TowerBackground>(topNode, "TowerInfoBackground_Sub2");
    if (!towerBkg_iter)
    {
      std::cout << "Aborting Run: Iter Background Info null" << std::endl;
      return Fun4AllReturnCodes::ABORTRUN;
    }

    m_data.is_flow_failure_iter = towerBkg_iter->get_flow_failure_flag();
    m_data.calo_v2_iter = towerBkg_iter->get_v2();
    m_data.seeds_iter = towerBkg_iter->get_nHIRecoSeedsSub();

    if (Verbosity() > 0)
    {
      std::cout << "JetValidationv3::process_UE - [Event " << event_id << "] [Iter] seeds: " << m_data.seeds_iter
                << " | v2: " << m_data.calo_v2_iter
                << " | flow_fail: " << m_data.is_flow_failure_iter << std::endl;
    }

    // debug: 1st iteration (seed type 0) seed count
    if (Verbosity() > 1)
    {
      TowerBackground* towerBkg_iter1 = findNode::getClass<TowerBackground>(topNode, "TowerInfoBackground_Sub1");
      if (towerBkg_iter1)
      {
        std::cout << "JetValidationv3::process_UE - [Event " << event_id << "] [Iter1] seeds: " << towerBkg_iter1->get_nHIRecoSeedsSub()
                  << " | v2: " << towerBkg_iter1->get_v2()
                  << " | flow_fail: " << towerBkg_iter1->get_flow_failure_flag() << std::endl;
      }
    }
  }

  if (m_do_iter_noflow)
  {
    TowerBackground* towerBkg_iternf = findNode::getClass<TowerBackground>(topNode, "TowerInfoBackground_NoFlowSub2");
    if (!towerBkg_iternf)
    {
      std::cout << "Aborting Run: Iter NoFlow Background Info null" << std::endl;
      return Fun4AllReturnCodes::ABORTRUN;
    }

    m_data.seeds_iternf = towerBkg_iternf->get_nHIRecoSeedsSub();

    if (Verbosity() > 0)
    {
      std::cout << "JetValidationv3::process_UE - [Event " << event_id << "] [IterNF] seeds: " << m_data.seeds_iternf << std::endl;
    }
  }

  if (m_do_mult)
  {
    TowerBackground* towerBkg_mult = findNode::getClass<TowerBackground>(topNode, "TowerInfoBackground_MultSub2");
    if (!towerBkg_mult)
    {
      std::cout << "Aborting Run: Mult Background Info null" << std::endl;
      return Fun4AllReturnCodes::ABORTRUN;
    }

    m_data.is_flow_failure_mult = towerBkg_mult->get_flow_failure_flag();
    m_data.calo_v2_mult = towerBkg_mult->get_v2();
    m_data.seeds_mult = towerBkg_mult->get_nHIRecoSeedsSub();

    if (Verbosity() > 0)
    {
      std::cout << "JetValidationv3::process_UE - [Event " << event_id << "] [Mult] seeds: " << m_data.seeds_mult
                << " | v2: " << m_data.calo_v2_mult
                << " | flow_fail: " << m_data.is_flow_failure_mult << std::endl;
    }
  }

  if (m_do_mult_noflow)
  {
    TowerBackground* towerBkg_multnf = findNode::getClass<TowerBackground>(topNode, "TowerInfoBackground_MultNoFlowSub2");
    if (!towerBkg_multnf)
    {
      std::cout << "Aborting Run: Mult NoFlow Background Info null" << std::endl;
      return Fun4AllReturnCodes::ABORTRUN;
    }

    m_data.seeds_multnf = towerBkg_multnf->get_nHIRecoSeedsSub();

    if (Verbosity() > 0)
    {
      std::cout << "JetValidationv3::process_UE - [Event " << event_id << "] [MultNF] seeds: " << m_data.seeds_multnf << std::endl;
    }
  }

  return Fun4AllReturnCodes::EVENT_OK;
}

//____________________________________________________________________________..
int JetValidationv3::process_jets(PHCompositeNode *topNode)
{
  EventHeader *eventInfo = findNode::getClass<EventHeader>(topNode, "EventHeader");
  int event_id = eventInfo ? eventInfo->get_EvtSequence() : -1;

  double zvtx = 0.0;
  GlobalVertexMap *vertexmap = findNode::getClass<GlobalVertexMap>(topNode, "GlobalVertexMap");
  if (!vertexmap)
  {
    std::cout << "JetValidationv3: GlobalVertexMap null" << std::endl;
  }
  else if (GlobalVertex *vtx = JetUtils::get_event_vertex(vertexmap))
  {
    zvtx = vtx->get_z();
  }

  // EMfrac-calibrated counterpart of a legacy calib node (nullptr unless m_do_emfrac)
  auto get_emfrac_jets = [&](const std::string& calib_node) -> JetContainer*
  {
    return m_do_emfrac ? findNode::getClass<JetContainer>(topNode, calib_node + m_emfrac_calib_suffix) : nullptr;
  };

  auto fill_jets = [&](JetContainer* jets, JetContainer* jets_calib, JetContainer* jets_calib_emfrac, JetData& jd, double jet_radius,
                       TowerInfoContainer* cemc_towers, TowerInfoContainer* ihcal_towers, TowerInfoContainer* ohcal_towers,
                       const std::string& jet_label = "")
  {
    if (Verbosity() > 1)
    {
      std::cout << "JetValidationv3::process_jets - [Event " << event_id << "] [" << jet_label << "] (R=" << jet_radius
                << ") Container sizes: raw=" << jets->size() << ", calib=" << jets_calib->size() << std::endl;
    }

    for (unsigned int i = 0; i < std::min(jets->size(), jets_calib->size()); ++i)
    {
      auto *jet = jets->get_jet(i);
      auto *jet_calib = jets_calib->get_jet(i);

      if (!jet || !jet_calib)
      {
        continue;
      }

      double pt = jet->get_pt();
      double pt_calib = jet_calib->get_pt();
      double energy = jet->get_e();
      double phi = jet->get_phi();
      double eta = jet->get_eta();

      // map [-pi,pi] -> [0,2pi]
      if (phi < 0)
      {
        phi += 2.0 * std::numbers::pi;
      }

      // with the EMfrac comparison, keep jets passing either calibration so both
      // populations are complete; the analysis must re-apply its own pT cut
      double pt_calib_emfrac = std::numeric_limits<double>::quiet_NaN();
      if (jets_calib_emfrac)
      {
        auto *jet_calib_emfrac = jets_calib_emfrac->get_jet(i);
        if (jet_calib_emfrac)
        {
          pt_calib_emfrac = jet_calib_emfrac->get_pt();
        }
      }

      bool pass_pt_legacy = (pt_calib >= m_jet_pt_min_cut);
      bool pass_pt = pass_pt_legacy || (pt_calib_emfrac >= m_jet_pt_min_cut);
      bool pass_eta = !JetUtils::check_bad_jet_eta(eta, zvtx, jet_radius);
      bool pass_cuts = pass_pt && pass_eta;

      if (Verbosity() > 2)
      {
        std::string status_str = "PASS";
        if (!pass_pt && !pass_eta)
        {
          status_str = "FAIL(pT+eta)";
        }
        else if (!pass_pt)
        {
          status_str = "FAIL(pT)";
        }
        else if (!pass_eta)
        {
          status_str = "FAIL(eta)";
        }

        std::cout << "JetValidationv3::process_jets - [Event " << event_id << "] [" << jet_label << "] Jet #" << i
                  << " | pT: " << pt << " -> pT_calib: " << pt_calib << " GeV"
                  << " | E: " << energy << " GeV"
                  << " | eta: " << eta
                  << " | phi: " << phi
                  << " | zvtx: " << zvtx << " cm"
                  << " | " << status_str
                  << std::endl;
      }

      if (pass_cuts)
      {
        jd.pt.push_back(pt);
        jd.pt_calib.push_back(pt_calib);
        jd.e.push_back(energy);
        jd.phi.push_back(phi);
        jd.eta.push_back(eta);

        if (m_do_emfrac)
        {
          jd.pt_calib_emfrac.push_back(pt_calib_emfrac);
        }

        if (m_do_emfrac || m_do_timing)
        {
          jd.emcal_frac.push_back(jet->get_emcal_frac());
          jd.ihcal_frac.push_back(jet->get_ihcal_frac());
          jd.ohcal_frac.push_back(jet->get_ohcal_frac());
        }

        if (energy > 0 && pass_pt_legacy)
        {
          jd.max_pt = std::max(jd.max_pt, pt_calib);
        }

        if (Verbosity() > 0 && Verbosity() <= 2)
        {
          std::cout << "JetValidationv3::process_jets - [Event " << event_id << "] [" << jet_label << "] Jet #" << i
                    << " | pT: " << pt << " -> pT_calib: " << pt_calib << " GeV"
                    << " | E: " << energy << " GeV"
                    << " | eta: " << eta
                    << " | phi: " << phi
                    << " | zvtx: " << zvtx << " cm"
                    << std::endl;
        }

        if (m_do_timing || m_do_detailed || m_do_lead_const)
        {
          // indexed by calo layer (see get_calo_layer)
          const std::array<TowerInfoContainer*, 3> calo_towers{cemc_towers, ihcal_towers, ohcal_towers};
          const std::array<const TowerGeomArray*, 3> calo_geom{&m_geom_cemc, &m_geom_hcalin, &m_geom_hcalout};
          const std::array<double, 3> calo_radius{m_r_cemc, m_r_hcalin, m_r_hcalout};

          // timing: energy-weighted tower time per calo layer, so the jet time is the energy-weighted mean of the three
          std::array<double, 3> t_sum{};
          std::array<double, 3> e_sum{};

          // leading constituent: highest vertex-corrected tower pT
          double lead_pt = std::numeric_limits<double>::quiet_NaN();
          double lead_e = std::numeric_limits<double>::quiet_NaN();
          int lead_calo = -1;
          int lead_index = -1;

          // detailed: constituent towers per calo layer
          std::array<std::vector<int>, 3> tower_idx;
          std::array<std::vector<double>, 3> tower_e;
          std::array<std::vector<double>, 3> tower_pt;

          for (const auto& [src, channel] : jet->get_comp_vec())
          {
            int calo = get_calo_layer(src);
            if (calo < 0)
            {
              continue;
            }

            TowerInfoContainer* towers = calo_towers[calo];
            if (!towers || channel >= towers->size())
            {
              continue;
            }

            TowerInfo *tower = towers->get_tower_at_channel(channel);
            if (!tower)
            {
              continue;
            }

            double tower_energy = tower->get_energy();

            // towers enter the jet time only above the timing threshold
            if (m_do_timing && tower_energy > m_timing_e_threshold)
            {
              t_sum[calo] += NS_PER_SAMPLE * tower->get_time() * tower_energy;
              e_sum[calo] += tower_energy;
            }

            if (!m_do_detailed && !m_do_lead_const)
            {
              continue;
            }

            unsigned int channelkey = towers->encode_key(channel);
            unsigned int ieta = towers->getTowerEtaBin(channelkey);
            unsigned int iphi = towers->getTowerPhiBin(channelkey);

            if (ieta >= CaloGeometry::HCAL_ETA_BINS || iphi >= CaloGeometry::HCAL_PHI_BINS)
            {
              continue;
            }

            size_t idx = static_cast<size_t>(ieta) * CaloGeometry::HCAL_PHI_BINS + iphi;
            const auto &geom = (*calo_geom[calo])[idx];
            if (!geom.is_valid)
            {
              continue;
            }

            double eta_corrected = geom.eta;
            if (zvtx != 0.0 && calo_radius[calo] > 0.0)
            {
              double z = geom.z0 - zvtx;
              eta_corrected = std::asinh(z / calo_radius[calo]);
            }

            double pt_tower = tower_energy / std::cosh(eta_corrected);

            if (lead_calo < 0 || pt_tower > lead_pt)
            {
              lead_pt = pt_tower;
              lead_e = tower_energy;
              lead_calo = calo;
              lead_index = static_cast<int>(channel);
            }

            if (m_do_detailed)
            {
              tower_idx[calo].push_back(static_cast<int>(channel));
              tower_e[calo].push_back(tower_energy);
              tower_pt[calo].push_back(pt_tower);
            }
          }

          if (m_do_timing)
          {
            // jet time: energy-weighted constituent tower time, set by FastJetAlgoSub
            // (NaN if no constituent tower was above the timing threshold)
            double t = (jet->size_properties() > Jet::PROPERTY::prop_t) ? jet->get_property(Jet::PROPERTY::prop_t)
                                                                         : std::numeric_limits<double>::quiet_NaN();

            auto layer_t = [&](int layer)
            {
              return e_sum[layer] > 0 ? t_sum[layer] / e_sum[layer] : std::numeric_limits<double>::quiet_NaN();
            };

            jd.t.push_back(t);
            jd.t_emcal.push_back(layer_t(0));
            jd.t_ihcal.push_back(layer_t(1));
            jd.t_ohcal.push_back(layer_t(2));

            if (Verbosity() > 2)
            {
              std::cout << "    Timing -> t: " << t << " ns | t_emcal: " << layer_t(0)
                        << " | t_ihcal: " << layer_t(1) << " | t_ohcal: " << layer_t(2) << std::endl;
            }
          }

          if (Verbosity() > 1 && (m_do_detailed || m_do_lead_const))
          {
            std::cout << "    Constituents -> EMCAL: " << tower_idx[0].size()
                      << " towers | IHCAL: " << tower_idx[1].size()
                      << " towers | OHCAL: " << tower_idx[2].size()
                      << " towers" << std::endl;
            std::cout << "    Leading constituent -> calo: " << lead_calo << " | index: " << lead_index
                      << " | pT: " << lead_pt << " GeV | E: " << lead_e << " GeV" << std::endl;
          }

          if (m_do_lead_const)
          {
            jd.lead_pt.push_back(lead_pt);
            jd.lead_e.push_back(lead_e);
            jd.lead_calo.push_back(lead_calo);
            jd.lead_index.push_back(lead_index);
          }

          if (m_do_detailed)
          {
            jd.emcal_tower_index.push_back(std::move(tower_idx[0]));
            jd.emcal_tower_energy.push_back(std::move(tower_e[0]));
            jd.emcal_tower_pt.push_back(std::move(tower_pt[0]));

            jd.ihcal_tower_index.push_back(std::move(tower_idx[1]));
            jd.ihcal_tower_energy.push_back(std::move(tower_e[1]));
            jd.ihcal_tower_pt.push_back(std::move(tower_pt[1]));

            jd.ohcal_tower_index.push_back(std::move(tower_idx[2]));
            jd.ohcal_tower_energy.push_back(std::move(tower_e[2]));
            jd.ohcal_tower_pt.push_back(std::move(tower_pt[2]));
          }
        }
      }
    }

    if (Verbosity() > 0 && !jd.pt.empty())
    {
      std::cout << "JetValidationv3::process_jets - [Event " << event_id << "] [" << jet_label << "] Total accepted jets: "
                << jd.pt.size() << " / " << std::min(jets->size(), jets_calib->size())
                << " (max pT_calib = " << jd.max_pt << " GeV/c)" << std::endl;
    }
  };

  if (m_do_iter)
  {
    TowerInfoContainer* cemc_sub1 = nullptr;
    TowerInfoContainer* ihcal_sub1 = nullptr;
    TowerInfoContainer* ohcal_sub1 = nullptr;
    if (m_do_detailed || m_do_timing || m_do_lead_const)
    {
      cemc_sub1 = findNode::getClass<TowerInfoContainer>(topNode, "TOWERINFO_CALIB_CEMC_RETOWER_SUB1");
      ihcal_sub1 = findNode::getClass<TowerInfoContainer>(topNode, "TOWERINFO_CALIB_HCALIN_SUB1");
      ohcal_sub1 = findNode::getClass<TowerInfoContainer>(topNode, "TOWERINFO_CALIB_HCALOUT_SUB1");
    }
    if ((m_do_timing || m_do_lead_const) && (!cemc_sub1 || !ihcal_sub1 || !ohcal_sub1))
    {
      std::cout << "Aborting Run: Iter Tower Info null" << std::endl;
      return Fun4AllReturnCodes::ABORTRUN;
    }

    if (m_do_r02)
    {
      JetContainer *jets_iter_r02 = findNode::getClass<JetContainer>(topNode, m_recoJetName_iter_r02);
      JetContainer *jets_iter_calib_r02 = findNode::getClass<JetContainer>(topNode, m_recoJetName_iter_calib_r02);
      JetContainer *jets_iter_calib_emfrac_r02 = get_emfrac_jets(m_recoJetName_iter_calib_r02);
      if (!jets_iter_r02 || !jets_iter_calib_r02 || (m_do_emfrac && !jets_iter_calib_emfrac_r02))
      {
        std::cout << "Aborting Run: Iter R=0.2 Jets Info null" << std::endl;
        return Fun4AllReturnCodes::ABORTRUN;
      }
      fill_jets(jets_iter_r02, jets_iter_calib_r02, jets_iter_calib_emfrac_r02, m_data.iter_r02, 0.2, cemc_sub1, ihcal_sub1, ohcal_sub1, "Iter R=0.2");
    }

    if (m_do_r03)
    {
      JetContainer *jets_iter_r03 = findNode::getClass<JetContainer>(topNode, m_recoJetName_iter_r03);
      JetContainer *jets_iter_calib_r03 = findNode::getClass<JetContainer>(topNode, m_recoJetName_iter_calib_r03);
      JetContainer *jets_iter_calib_emfrac_r03 = get_emfrac_jets(m_recoJetName_iter_calib_r03);
      if (!jets_iter_r03 || !jets_iter_calib_r03 || (m_do_emfrac && !jets_iter_calib_emfrac_r03))
      {
        std::cout << "Aborting Run: Iter R=0.3 Jets Info null" << std::endl;
        return Fun4AllReturnCodes::ABORTRUN;
      }
      fill_jets(jets_iter_r03, jets_iter_calib_r03, jets_iter_calib_emfrac_r03, m_data.iter_r03, 0.3, cemc_sub1, ihcal_sub1, ohcal_sub1, "Iter R=0.3");
    }
  }

  if (m_do_iter_noflow)
  {
    TowerInfoContainer* cemc_iternf = nullptr;
    TowerInfoContainer* ihcal_iternf = nullptr;
    TowerInfoContainer* ohcal_iternf = nullptr;
    if (m_do_detailed || m_do_timing || m_do_lead_const)
    {
      cemc_iternf = findNode::getClass<TowerInfoContainer>(topNode, m_towerNode_iternf_cemc);
      ihcal_iternf = findNode::getClass<TowerInfoContainer>(topNode, m_towerNode_iternf_ihcal);
      ohcal_iternf = findNode::getClass<TowerInfoContainer>(topNode, m_towerNode_iternf_ohcal);
    }
    if ((m_do_timing || m_do_lead_const) && (!cemc_iternf || !ihcal_iternf || !ohcal_iternf))
    {
      std::cout << "Aborting Run: Iter NoFlow Tower Info null" << std::endl;
      return Fun4AllReturnCodes::ABORTRUN;
    }

    if (m_do_r02)
    {
      JetContainer *jets_iternf_r02 = findNode::getClass<JetContainer>(topNode, m_recoJetName_iternf_r02);
      JetContainer *jets_iternf_calib_r02 = findNode::getClass<JetContainer>(topNode, m_recoJetName_iternf_calib_r02);
      JetContainer *jets_iternf_calib_emfrac_r02 = get_emfrac_jets(m_recoJetName_iternf_calib_r02);
      if (!jets_iternf_r02 || !jets_iternf_calib_r02 || (m_do_emfrac && !jets_iternf_calib_emfrac_r02))
      {
        std::cout << "Aborting Run: Iter NoFlow R=0.2 Jets Info null" << std::endl;
        return Fun4AllReturnCodes::ABORTRUN;
      }
      fill_jets(jets_iternf_r02, jets_iternf_calib_r02, jets_iternf_calib_emfrac_r02, m_data.iternf_r02, 0.2, cemc_iternf, ihcal_iternf, ohcal_iternf, "IterNF R=0.2");
    }

    if (m_do_r03)
    {
      JetContainer *jets_iternf_r03 = findNode::getClass<JetContainer>(topNode, m_recoJetName_iternf_r03);
      JetContainer *jets_iternf_calib_r03 = findNode::getClass<JetContainer>(topNode, m_recoJetName_iternf_calib_r03);
      JetContainer *jets_iternf_calib_emfrac_r03 = get_emfrac_jets(m_recoJetName_iternf_calib_r03);
      if (!jets_iternf_r03 || !jets_iternf_calib_r03 || (m_do_emfrac && !jets_iternf_calib_emfrac_r03))
      {
        std::cout << "Aborting Run: Iter NoFlow R=0.3 Jets Info null" << std::endl;
        return Fun4AllReturnCodes::ABORTRUN;
      }
      fill_jets(jets_iternf_r03, jets_iternf_calib_r03, jets_iternf_calib_emfrac_r03, m_data.iternf_r03, 0.3, cemc_iternf, ihcal_iternf, ohcal_iternf, "IterNF R=0.3");
    }
  }

  if (m_do_mult)
  {
    TowerInfoContainer* cemc_mult = nullptr;
    TowerInfoContainer* ihcal_mult = nullptr;
    TowerInfoContainer* ohcal_mult = nullptr;
    if (m_do_detailed || m_do_timing || m_do_lead_const)
    {
      cemc_mult = findNode::getClass<TowerInfoContainer>(topNode, m_towerNode_mult_cemc);
      ihcal_mult = findNode::getClass<TowerInfoContainer>(topNode, m_towerNode_mult_ihcal);
      ohcal_mult = findNode::getClass<TowerInfoContainer>(topNode, m_towerNode_mult_ohcal);
    }
    if ((m_do_timing || m_do_lead_const) && (!cemc_mult || !ihcal_mult || !ohcal_mult))
    {
      std::cout << "Aborting Run: Mult Tower Info null" << std::endl;
      return Fun4AllReturnCodes::ABORTRUN;
    }

    if (m_do_r02)
    {
      JetContainer *jets_mult_r02 = findNode::getClass<JetContainer>(topNode, m_recoJetName_mult_r02);
      JetContainer *jets_mult_calib_r02 = findNode::getClass<JetContainer>(topNode, m_recoJetName_mult_calib_r02);
      JetContainer *jets_mult_calib_emfrac_r02 = get_emfrac_jets(m_recoJetName_mult_calib_r02);
      if (!jets_mult_r02 || !jets_mult_calib_r02 || (m_do_emfrac && !jets_mult_calib_emfrac_r02))
      {
        std::cout << "Aborting Run: Mult R=0.2 Jets Info null" << std::endl;
        return Fun4AllReturnCodes::ABORTRUN;
      }
      fill_jets(jets_mult_r02, jets_mult_calib_r02, jets_mult_calib_emfrac_r02, m_data.mult_r02, 0.2, cemc_mult, ihcal_mult, ohcal_mult, "Mult R=0.2");
    }

    if (m_do_r03)
    {
      JetContainer *jets_mult_r03 = findNode::getClass<JetContainer>(topNode, m_recoJetName_mult_r03);
      JetContainer *jets_mult_calib_r03 = findNode::getClass<JetContainer>(topNode, m_recoJetName_mult_calib_r03);
      JetContainer *jets_mult_calib_emfrac_r03 = get_emfrac_jets(m_recoJetName_mult_calib_r03);
      if (!jets_mult_r03 || !jets_mult_calib_r03 || (m_do_emfrac && !jets_mult_calib_emfrac_r03))
      {
        std::cout << "Aborting Run: Mult R=0.3 Jets Info null" << std::endl;
        return Fun4AllReturnCodes::ABORTRUN;
      }
      fill_jets(jets_mult_r03, jets_mult_calib_r03, jets_mult_calib_emfrac_r03, m_data.mult_r03, 0.3, cemc_mult, ihcal_mult, ohcal_mult, "Mult R=0.3");
    }
  }

  if (m_do_mult_noflow)
  {
    TowerInfoContainer* cemc_multnf = nullptr;
    TowerInfoContainer* ihcal_multnf = nullptr;
    TowerInfoContainer* ohcal_multnf = nullptr;
    if (m_do_detailed || m_do_timing || m_do_lead_const)
    {
      cemc_multnf = findNode::getClass<TowerInfoContainer>(topNode, m_towerNode_multnf_cemc);
      ihcal_multnf = findNode::getClass<TowerInfoContainer>(topNode, m_towerNode_multnf_ihcal);
      ohcal_multnf = findNode::getClass<TowerInfoContainer>(topNode, m_towerNode_multnf_ohcal);
    }
    if ((m_do_timing || m_do_lead_const) && (!cemc_multnf || !ihcal_multnf || !ohcal_multnf))
    {
      std::cout << "Aborting Run: Mult NoFlow Tower Info null" << std::endl;
      return Fun4AllReturnCodes::ABORTRUN;
    }

    if (m_do_r02)
    {
      JetContainer *jets_multnf_r02 = findNode::getClass<JetContainer>(topNode, m_recoJetName_multnf_r02);
      JetContainer *jets_multnf_calib_r02 = findNode::getClass<JetContainer>(topNode, m_recoJetName_multnf_calib_r02);
      JetContainer *jets_multnf_calib_emfrac_r02 = get_emfrac_jets(m_recoJetName_multnf_calib_r02);
      if (!jets_multnf_r02 || !jets_multnf_calib_r02 || (m_do_emfrac && !jets_multnf_calib_emfrac_r02))
      {
        std::cout << "Aborting Run: Mult NoFlow R=0.2 Jets Info null" << std::endl;
        return Fun4AllReturnCodes::ABORTRUN;
      }
      fill_jets(jets_multnf_r02, jets_multnf_calib_r02, jets_multnf_calib_emfrac_r02, m_data.multnf_r02, 0.2, cemc_multnf, ihcal_multnf, ohcal_multnf, "MultNF R=0.2");
    }

    if (m_do_r03)
    {
      JetContainer *jets_multnf_r03 = findNode::getClass<JetContainer>(topNode, m_recoJetName_multnf_r03);
      JetContainer *jets_multnf_calib_r03 = findNode::getClass<JetContainer>(topNode, m_recoJetName_multnf_calib_r03);
      JetContainer *jets_multnf_calib_emfrac_r03 = get_emfrac_jets(m_recoJetName_multnf_calib_r03);
      if (!jets_multnf_r03 || !jets_multnf_calib_r03 || (m_do_emfrac && !jets_multnf_calib_emfrac_r03))
      {
        std::cout << "Aborting Run: Mult NoFlow R=0.3 Jets Info null" << std::endl;
        return Fun4AllReturnCodes::ABORTRUN;
      }
      fill_jets(jets_multnf_r03, jets_multnf_calib_r03, jets_multnf_calib_emfrac_r03, m_data.multnf_r03, 0.3, cemc_multnf, ihcal_multnf, ohcal_multnf, "MultNF R=0.3");
    }
  }

  if (m_do_unsub)
  {
    TowerInfoContainer* cemc_unsub = nullptr;
    TowerInfoContainer* ihcal_unsub = nullptr;
    TowerInfoContainer* ohcal_unsub = nullptr;
    if (m_do_detailed || m_do_timing || m_do_lead_const)
    {
      cemc_unsub = findNode::getClass<TowerInfoContainer>(topNode, "TOWERINFO_CALIB_CEMC_RETOWER");
      ihcal_unsub = findNode::getClass<TowerInfoContainer>(topNode, "TOWERINFO_CALIB_HCALIN");
      ohcal_unsub = findNode::getClass<TowerInfoContainer>(topNode, "TOWERINFO_CALIB_HCALOUT");
    }
    if ((m_do_timing || m_do_lead_const) && (!cemc_unsub || !ihcal_unsub || !ohcal_unsub))
    {
      std::cout << "Aborting Run: Unsub Tower Info null" << std::endl;
      return Fun4AllReturnCodes::ABORTRUN;
    }

    if (m_do_r02)
    {
      JetContainer *jets_unsub_r02 = findNode::getClass<JetContainer>(topNode, m_recoJetName_unsub_r02);
      JetContainer *jets_unsub_calib_r02 = findNode::getClass<JetContainer>(topNode, m_recoJetName_unsub_calib_r02);
      JetContainer *jets_unsub_calib_emfrac_r02 = get_emfrac_jets(m_recoJetName_unsub_calib_r02);
      if (!jets_unsub_r02 || !jets_unsub_calib_r02 || (m_do_emfrac && !jets_unsub_calib_emfrac_r02))
      {
        std::cout << "Aborting Run: Unsub R=0.2 Jets Info null" << std::endl;
        return Fun4AllReturnCodes::ABORTRUN;
      }
      fill_jets(jets_unsub_r02, jets_unsub_calib_r02, jets_unsub_calib_emfrac_r02, m_data.unsub_r02, 0.2, cemc_unsub, ihcal_unsub, ohcal_unsub, "Unsub R=0.2");
    }

    if (m_do_r03)
    {
      JetContainer *jets_unsub_r03 = findNode::getClass<JetContainer>(topNode, m_recoJetName_unsub_r03);
      JetContainer *jets_unsub_calib_r03 = findNode::getClass<JetContainer>(topNode, m_recoJetName_unsub_calib_r03);
      JetContainer *jets_unsub_calib_emfrac_r03 = get_emfrac_jets(m_recoJetName_unsub_calib_r03);
      if (!jets_unsub_r03 || !jets_unsub_calib_r03 || (m_do_emfrac && !jets_unsub_calib_emfrac_r03))
      {
        std::cout << "Aborting Run: Unsub R=0.3 Jets Info null" << std::endl;
        return Fun4AllReturnCodes::ABORTRUN;
      }
      fill_jets(jets_unsub_r03, jets_unsub_calib_r03, jets_unsub_calib_emfrac_r03, m_data.unsub_r03, 0.3, cemc_unsub, ihcal_unsub, ohcal_unsub, "Unsub R=0.3");
    }
  }

  return Fun4AllReturnCodes::EVENT_OK;
}

//____________________________________________________________________________..
int JetValidationv3::process_event(PHCompositeNode *topNode)
{
  if (m_do_iter || m_do_iter_noflow || m_do_mult || m_do_mult_noflow)
  {
    int ret = process_UE(topNode);
    if (ret)
    {
      return ret;
    }
  }

  if (m_do_iter || m_do_iter_noflow || m_do_mult || m_do_mult_noflow || m_do_unsub)
  {
    int ret = process_jets(topNode);
    if (ret)
    {
      return ret;
    }
  }

  // TTree filling is managed by TreeFiller

  return Fun4AllReturnCodes::EVENT_OK;
}

//____________________________________________________________________________..
int JetValidationv3::ResetEvent([[maybe_unused]] PHCompositeNode *topNode)
{
  // UE
  if (m_do_iter)
  {
    m_data.seeds_iter = 0;
    m_data.calo_v2_iter = 9999;
    m_data.is_flow_failure_iter = false;
    if (m_do_r02)
    {
      m_data.iter_r02.clear();
    }
    if (m_do_r03)
    {
      m_data.iter_r03.clear();
    }
  }

  if (m_do_iter_noflow)
  {
    m_data.seeds_iternf = 0;
    if (m_do_r02)
    {
      m_data.iternf_r02.clear();
    }
    if (m_do_r03)
    {
      m_data.iternf_r03.clear();
    }
  }

  if (m_do_mult)
  {
    m_data.seeds_mult = 0;
    m_data.calo_v2_mult = 9999;
    m_data.is_flow_failure_mult = false;
    if (m_do_r02)
    {
      m_data.mult_r02.clear();
    }
    if (m_do_r03)
    {
      m_data.mult_r03.clear();
    }
  }

  if (m_do_mult_noflow)
  {
    m_data.seeds_multnf = 0;
    if (m_do_r02)
    {
      m_data.multnf_r02.clear();
    }
    if (m_do_r03)
    {
      m_data.multnf_r03.clear();
    }
  }

  if (m_do_unsub)
  {
    if (m_do_r02)
    {
      m_data.unsub_r02.clear();
    }
    if (m_do_r03)
    {
      m_data.unsub_r03.clear();
    }
  }

  return Fun4AllReturnCodes::EVENT_OK;
}

//____________________________________________________________________________..
int JetValidationv3::End([[maybe_unused]] PHCompositeNode *topNode)
{
  std::cout << "JetValidationv3::End" << std::endl;
  return Fun4AllReturnCodes::EVENT_OK;
}
