#include "GlobalQA.h"

// -- Fun4All
#include <ffaobjects/EventHeader.h>
#include <fun4all/Fun4AllReturnCodes.h>
#include <fun4all/Fun4AllServer.h>
#include <phool/PHCompositeNode.h>
#include <phool/getClass.h>

// -- Event Plane
#include <eventplaneinfo/EventplaneinfoMap.h>
#include <eventplaneinfo/Eventplaneinfo.h>

// -- Calo
#include <calobase/TowerInfo.h>
#include <calobase/TowerInfoContainer.h>
#include <calobase/TowerInfoDefs.h>

// -- MBD / MinBias
#include <pdbcalbase/PdbParameterMap.h>
#include <phparameter/PHParameters.h>

// -- Centrality
#include <centrality/CentralityInfo.h>

// -- ROOT
#include <TH1F.h>
#include <TH2F.h>
#include <TProfile.h>
#include <TProfile2D.h>
#include <TTree.h>

#include <treefiller/TreeFiller.h>

#include <cmath>
#include <format>
#include <iostream>
#include <limits>
#include <numbers>

GlobalQA::GlobalQA(const std::string &name)
  : SubsysReco(name)
{
}

int GlobalQA::Init([[maybe_unused]] PHCompositeNode *topNode)
{
  if (m_do_tree)
  {
    TTree *tree = TreeFiller::getTree();
    if (tree)
    {
      if (m_do_ep)
      {
        // sEPD - Event Plane
        tree->Branch("psi2_raw_S", &m_data.psi2_raw_S);
        tree->Branch("psi2_raw_N", &m_data.psi2_raw_N);
        tree->Branch("psi2_raw_NS", &m_data.psi2_raw_NS);

        tree->Branch("psi2_S", &m_data.psi2_S);
        tree->Branch("psi2_N", &m_data.psi2_N);
        tree->Branch("psi2_NS", &m_data.psi2_NS);
      }

      if (m_do_sepd)
      {
        // sEPD - QA
        tree->Branch("sepd_charge_south", &m_data.sepd_charge_south);
        tree->Branch("sepd_charge_north", &m_data.sepd_charge_north);
      }

      if (m_do_mbd)
      {
        // MBD - QA
        tree->Branch("mbd_charge_south", &m_data.mbd_charge_south);
        tree->Branch("mbd_charge_north", &m_data.mbd_charge_north);
      }
    }
  }

  if (m_do_hist)
  {
    Fun4AllServer *se = Fun4AllServer::instance();

    if (m_do_sepd)
    {
      m_hists.hSEPD_Charge_Total = new TH1F("hSEPD_Charge_Total", "sEPD Total Charge; Total Charge [MIP]; Counts",
                                            m_hist_config.bins_sepd_charge, m_hist_config.sepd_charge_total_low, m_hist_config.sepd_charge_total_high);
      se->registerHisto(m_hists.hSEPD_Charge_Total);

      m_hists.hSEPD_Hits_Total = new TH1F("hSEPD_Hits_Total", "sEPD Total Hit Multiplicity; Total Hits; Counts",
                                          m_hist_config.bins_sepd_hits_total, m_hist_config.sepd_hits_total_low, m_hist_config.sepd_hits_total_high);
      se->registerHisto(m_hists.hSEPD_Hits_Total);

      m_hists.h2SEPD_Charge_South_North = new TH2F("h2SEPD_Charge_South_North", "sEPD Charge Correlation; South Charge [MIP]; North Charge [MIP]",
                                                   m_hist_config.bins_sepd_charge, m_hist_config.sepd_charge_arm_low, m_hist_config.sepd_charge_arm_high,
                                                   m_hist_config.bins_sepd_charge, m_hist_config.sepd_charge_arm_low, m_hist_config.sepd_charge_arm_high);
      se->registerHisto(m_hists.h2SEPD_Charge_South_North);

      m_hists.h2SEPD_Hits_South_North = new TH2F("h2SEPD_Hits_South_North", "sEPD Hit Multiplicity Correlation; South Hits; North Hits",
                                                 m_hist_config.bins_sepd_hits_arm, m_hist_config.sepd_hits_arm_low, m_hist_config.sepd_hits_arm_high,
                                                 m_hist_config.bins_sepd_hits_arm, m_hist_config.sepd_hits_arm_low, m_hist_config.sepd_hits_arm_high);
      se->registerHisto(m_hists.h2SEPD_Hits_South_North);

      m_hists.h2SEPD_Channel_Charge = new TH2F("h2SEPD_Channel_Charge", "sEPD Channel vs Charge; Channel Index; Charge [MIP]",
                                               m_hist_config.bins_sepd_channels, m_hist_config.sepd_channel_low, m_hist_config.sepd_channel_high,
                                               m_hist_config.bins_sepd_channel_charge, m_hist_config.sepd_channel_charge_low, m_hist_config.sepd_channel_charge_high);
      se->registerHisto(m_hists.h2SEPD_Channel_Charge);

      m_hists.pSEPD_Channel_Charge = new TProfile("pSEPD_Channel_Charge", "sEPD Mean Charge per Channel; Channel Index; #LTCharge#GT [MIP]",
                                                 m_hist_config.bins_sepd_channels, m_hist_config.sepd_channel_low, m_hist_config.sepd_channel_high);
      se->registerHisto(m_hists.pSEPD_Channel_Charge);

      m_hists.hSEPD_Channel_NonFinite = new TH1F("hSEPD_Channel_NonFinite", "sEPD Non-finite Channel Charge (skipped); Channel Index; Counts",
                                                 m_hist_config.bins_sepd_channels, m_hist_config.sepd_channel_low, m_hist_config.sepd_channel_high);
      se->registerHisto(m_hists.hSEPD_Channel_NonFinite);

      m_hists.pSEPD_Ring_Charge_South = new TProfile("pSEPD_Ring_Charge_South", "sEPD South Mean Charge per Ring; Radial Bin; #LTCharge#GT [MIP]",
                                                     m_hist_config.bins_sepd_rings, m_hist_config.sepd_ring_low, m_hist_config.sepd_ring_high);
      se->registerHisto(m_hists.pSEPD_Ring_Charge_South);

      m_hists.pSEPD_Ring_Charge_North = new TProfile("pSEPD_Ring_Charge_North", "sEPD North Mean Charge per Ring; Radial Bin; #LTCharge#GT [MIP]",
                                                     m_hist_config.bins_sepd_rings, m_hist_config.sepd_ring_low, m_hist_config.sepd_ring_high);
      se->registerHisto(m_hists.pSEPD_Ring_Charge_North);

      // (ring, phi) maps per arm: dead/hot channel check
      m_hists.hSEPD_NEvents = new TH1F("hSEPD_NEvents", "sEPD Events Processed (occupancy normalization); ; Events", 1, 0, 1);
      se->registerHisto(m_hists.hSEPD_NEvents);

      const std::array<std::string, kNArm> arm_names = {"South", "North"};
      for (unsigned int arm = 0; arm < kNArm; ++arm)
      {
        const std::string &arm_name = arm_names[arm];

        m_hists.h2SEPD_RingPhi_Hits[arm] = new TH2F(std::format("h2SEPD_RingPhi_Hits_{}", arm_name).c_str(),
                                                    std::format("sEPD {} Hits (charge #geq threshold); Ring; #phi Bin; Hits", arm_name).c_str(),
                                                    m_hist_config.bins_sepd_rings, m_hist_config.sepd_ring_low, m_hist_config.sepd_ring_high,
                                                    m_hist_config.bins_sepd_phi, m_hist_config.sepd_phi_low, m_hist_config.sepd_phi_high);
        se->registerHisto(m_hists.h2SEPD_RingPhi_Hits[arm]);

        m_hists.pSEPD_RingPhi_Charge[arm] = new TProfile2D(std::format("pSEPD_RingPhi_Charge_{}", arm_name).c_str(),
                                                           std::format("sEPD {} Mean Hit Charge; Ring; #phi Bin; #LTCharge#GT [MIP]", arm_name).c_str(),
                                                           m_hist_config.bins_sepd_rings, m_hist_config.sepd_ring_low, m_hist_config.sepd_ring_high,
                                                           m_hist_config.bins_sepd_phi, m_hist_config.sepd_phi_low, m_hist_config.sepd_phi_high);
        se->registerHisto(m_hists.pSEPD_RingPhi_Charge[arm]);
      }

      m_hists.h2SEPD_Charge_Centrality = new TH2F("h2SEPD_Charge_Centrality", "sEPD Total Charge vs Centrality; Centrality [%]; sEPD Total Charge [MIP]",
                                                  m_hist_config.bins_cent, m_hist_config.cent_low, m_hist_config.cent_high,
                                                  m_hist_config.bins_sepd_charge, m_hist_config.sepd_charge_total_low, m_hist_config.sepd_charge_total_high);
      se->registerHisto(m_hists.h2SEPD_Charge_Centrality);
    }

    if (m_do_mbd)
    {
      m_hists.hMBD_Charge_Total = new TH1F("hMBD_Charge_Total", "MBD Total Charge; Total Charge; Counts",
                                           m_hist_config.bins_mbd_charge, m_hist_config.mbd_charge_total_low, m_hist_config.mbd_charge_total_high);
      se->registerHisto(m_hists.hMBD_Charge_Total);

      m_hists.h2MBD_Charge_South_North = new TH2F("h2MBD_Charge_South_North", "MBD Charge Correlation; South Charge; North Charge",
                                                  m_hist_config.bins_mbd_charge, m_hist_config.mbd_charge_arm_low, m_hist_config.mbd_charge_arm_high,
                                                  m_hist_config.bins_mbd_charge, m_hist_config.mbd_charge_arm_low, m_hist_config.mbd_charge_arm_high);
      se->registerHisto(m_hists.h2MBD_Charge_South_North);
    }

    if (m_do_sepd && m_do_mbd)
    {
      m_hists.h2SEPD_MBD_Total_Charge = new TH2F("h2SEPD_MBD_Total_Charge", "sEPD vs MBD Total Charge; MBD Total Charge; sEPD Total Charge [MIP]",
                                                 m_hist_config.bins_mbd_charge, m_hist_config.mbd_charge_total_low, m_hist_config.mbd_charge_total_high,
                                                 m_hist_config.bins_sepd_charge, m_hist_config.sepd_charge_total_low, m_hist_config.sepd_charge_total_high);
      se->registerHisto(m_hists.h2SEPD_MBD_Total_Charge);
    }

    if (m_do_ep)
    {
      // 2D South vs North correlations (projections give South/North 1D angles)
      m_hists.h2Psi2_raw_SN = new TH2F("h2Psi2_raw_SN", "sEPD Raw 2#Psi_{2} South vs North; 2#Psi_{2}^{raw} South [rad]; 2#Psi_{2}^{raw} North [rad]",
                                       m_hist_config.bins_psi, m_hist_config.psi_low, m_hist_config.psi_high,
                                       m_hist_config.bins_psi, m_hist_config.psi_low, m_hist_config.psi_high);
      se->registerHisto(m_hists.h2Psi2_raw_SN);

      m_hists.h2Psi2_SN = new TH2F("h2Psi2_SN", "sEPD Calibrated 2#Psi_{2} South vs North; 2#Psi_{2} South [rad]; 2#Psi_{2} North [rad]",
                                   m_hist_config.bins_psi, m_hist_config.psi_low, m_hist_config.psi_high,
                                   m_hist_config.bins_psi, m_hist_config.psi_low, m_hist_config.psi_high);
      se->registerHisto(m_hists.h2Psi2_SN);

      // Delta Psi2 and resolution
      m_hists.hDeltaPsi2_SN = new TH1F("hDeltaPsi2_SN", "sEPD Calibrated #Delta(2#Psi_{2}) South - North; 2#Psi_{2}^{S} - 2#Psi_{2}^{N} [rad]; Counts",
                                       m_hist_config.bins_psi, m_hist_config.psi_low, m_hist_config.psi_high);
      se->registerHisto(m_hists.hDeltaPsi2_SN);

      m_hists.hCos2DeltaPsi2 = new TH1F("hCos2DeltaPsi2", "sEPD cos(2(#Psi_{2}^{S} - #Psi_{2}^{N})); cos(2#Delta#Psi_{2}); Counts",
                                        100, -1.0, 1.0);
      se->registerHisto(m_hists.hCos2DeltaPsi2);

      // Event Plane vs Centrality (matching Jet-Anav3.C / plot_jet_bkgsub_qa.py)
      m_hists.h2Psi2_S_raw = new TH2F("h2Psi2_S_raw", "sEPD Raw 2#Psi_{2} South vs Centrality; 2#Psi_{2}^{raw} [rad]; Centrality [%]",
                                      m_hist_config.bins_psi, m_hist_config.psi_low, m_hist_config.psi_high,
                                      m_hist_config.bins_cent, m_hist_config.cent_low, m_hist_config.cent_high);
      se->registerHisto(m_hists.h2Psi2_S_raw);

      m_hists.h2Psi2_N_raw = new TH2F("h2Psi2_N_raw", "sEPD Raw 2#Psi_{2} North vs Centrality; 2#Psi_{2}^{raw} [rad]; Centrality [%]",
                                      m_hist_config.bins_psi, m_hist_config.psi_low, m_hist_config.psi_high,
                                      m_hist_config.bins_cent, m_hist_config.cent_low, m_hist_config.cent_high);
      se->registerHisto(m_hists.h2Psi2_N_raw);

      m_hists.h2Psi2_NS_raw = new TH2F("h2Psi2_NS_raw", "sEPD Raw 2#Psi_{2} Combined vs Centrality; 2#Psi_{2}^{raw} [rad]; Centrality [%]",
                                       m_hist_config.bins_psi, m_hist_config.psi_low, m_hist_config.psi_high,
                                       m_hist_config.bins_cent, m_hist_config.cent_low, m_hist_config.cent_high);
      se->registerHisto(m_hists.h2Psi2_NS_raw);

      m_hists.h2Psi2_S = new TH2F("h2Psi2_S", "sEPD Calibrated 2#Psi_{2} South vs Centrality; 2#Psi_{2} [rad]; Centrality [%]",
                                  m_hist_config.bins_psi, m_hist_config.psi_low, m_hist_config.psi_high,
                                  m_hist_config.bins_cent, m_hist_config.cent_low, m_hist_config.cent_high);
      se->registerHisto(m_hists.h2Psi2_S);

      m_hists.h2Psi2_N = new TH2F("h2Psi2_N", "sEPD Calibrated 2#Psi_{2} North vs Centrality; 2#Psi_{2} [rad]; Centrality [%]",
                                  m_hist_config.bins_psi, m_hist_config.psi_low, m_hist_config.psi_high,
                                  m_hist_config.bins_cent, m_hist_config.cent_low, m_hist_config.cent_high);
      se->registerHisto(m_hists.h2Psi2_N);

      m_hists.h2Psi2_NS = new TH2F("h2Psi2_NS", "sEPD Calibrated 2#Psi_{2} Combined vs Centrality; 2#Psi_{2} [rad]; Centrality [%]",
                                   m_hist_config.bins_psi, m_hist_config.psi_low, m_hist_config.psi_high,
                                   m_hist_config.bins_cent, m_hist_config.cent_low, m_hist_config.cent_high);
      se->registerHisto(m_hists.h2Psi2_NS);

      m_hists.pCos2DeltaPsi2_Cent = new TProfile("pCos2DeltaPsi2_Cent", "sEPD Sub-event Resolution vs Centrality; Centrality [%]; #LTcos(2#Delta#Psi_{2})#GT",
                                                m_hist_config.bins_cent_profile, m_hist_config.cent_low, m_hist_config.cent_high);
      se->registerHisto(m_hists.pCos2DeltaPsi2_Cent);

      m_hists.pSin2DeltaPsi2_Cent = new TProfile("pSin2DeltaPsi2_Cent", "sEPD #LTsin(2#Delta#Psi_{2})#GT vs Centrality (expect 0); Centrality [%]; #LTsin(2(#Psi_{2}^{S} - #Psi_{2}^{N}))#GT",
                                                 m_hist_config.bins_cent_profile, m_hist_config.cent_low, m_hist_config.cent_high);
      se->registerHisto(m_hists.pSin2DeltaPsi2_Cent);

      m_hists.pCos3DeltaPsi3_Cent = new TProfile("pCos3DeltaPsi3_Cent", "sEPD n = 3 Sub-event Resolution vs Centrality; Centrality [%]; #LTcos(3(#Psi_{3}^{S} - #Psi_{3}^{N}))#GT",
                                                 m_hist_config.bins_cent_profile, m_hist_config.cent_low, m_hist_config.cent_high);
      se->registerHisto(m_hists.pCos3DeltaPsi3_Cent);

      // calibration status: the Psi_2 histograms above (except the raw ones) hold calibrated events only
      m_hists.hEP_Status = new TH1F("hEP_Status", "sEPD Event Plane Calibration Status; ; Events", EP_NSTATUS, -0.5, static_cast<double>(EP_NSTATUS) - 0.5);
      m_hists.hEP_Status->GetXaxis()->SetBinLabel(EP_CALIBRATED + 1, "Calibrated");
      m_hists.hEP_Status->GetXaxis()->SetBinLabel(EP_ZERO_CHARGE + 1, "Zero Charge");
      m_hists.hEP_Status->GetXaxis()->SetBinLabel(EP_CENT_OUT_RANGE + 1, std::format("Cent #notin [0, {:.0f})", m_ep_calib_cent_max).c_str());
      m_hists.hEP_Status->GetXaxis()->SetBinLabel(EP_NO_CALIB + 1, "No Calib");
      m_hists.hEP_Status->GetXaxis()->SetBinLabel(EP_NON_FINITE + 1, "Non-finite Q");
      se->registerHisto(m_hists.hEP_Status);

      // Q_2 moments per calibration stage, 1% centrality bins as in the calibration
      const std::array<std::string, kNEPDet> det_names = {"S", "N", "NS"};
      const std::array<std::string, kNQStage> stage_names = {"raw", "rec", "flat"};
      const std::array<std::string, kNQStage> stage_titles = {"Raw", "Recentered", "Flattened"};
      const std::array<std::string, kNQMoment> moment_names = {"x", "y", "xx", "yy", "xy"};
      const std::array<std::string, kNQMoment> moment_titles = {"Q_{x}", "Q_{y}", "Q_{x}^{2}", "Q_{y}^{2}", "Q_{x}Q_{y}"};

      for (unsigned int stage = 0; stage < kNQStage; ++stage)
      {
        for (unsigned int det = 0; det < kNEPDet; ++det)
        {
          for (unsigned int mom = 0; mom < kNQMoment; ++mom)
          {
            auto *&prof = m_hists.pQ2_Moment[stage][det][mom];
            prof = new TProfile(std::format("pQ2_{}_{}_{}", moment_names[mom], stage_names[stage], det_names[det]).c_str(),
                                std::format("sEPD {} {} n = 2; Centrality [%]; #LT{}#GT", det_names[det], stage_titles[stage], moment_titles[mom]).c_str(),
                                m_hist_config.bins_cent, m_hist_config.cent_low, m_hist_config.cent_high);
            se->registerHisto(prof);
          }
        }
      }

      // Fourier flatness of the calibrated Psi_2
      for (unsigned int det = 0; det < kNEPDet; ++det)
      {
        for (unsigned int k = 1; k <= kNFlatTerms; ++k)
        {
          m_hists.pFlat_Cos[det][k - 1] = new TProfile(std::format("pFlat_Cos{}Psi2_{}", 2 * k, det_names[det]).c_str(),
                                                       std::format("sEPD {} Calibrated Flatness (expect 0); Centrality [%]; #LTcos({}#Psi_{{2}})#GT", det_names[det], 2 * k).c_str(),
                                                       m_hist_config.bins_cent, m_hist_config.cent_low, m_hist_config.cent_high);
          se->registerHisto(m_hists.pFlat_Cos[det][k - 1]);

          m_hists.pFlat_Sin[det][k - 1] = new TProfile(std::format("pFlat_Sin{}Psi2_{}", 2 * k, det_names[det]).c_str(),
                                                       std::format("sEPD {} Calibrated Flatness (expect 0); Centrality [%]; #LTsin({}#Psi_{{2}})#GT", det_names[det], 2 * k).c_str(),
                                                       m_hist_config.bins_cent, m_hist_config.cent_low, m_hist_config.cent_high);
          se->registerHisto(m_hists.pFlat_Sin[det][k - 1]);
        }
      }
    }
  }

  return Fun4AllReturnCodes::EVENT_OK;
}

//____________________________________________________________________________..
int GlobalQA::process_event(PHCompositeNode *topNode)
{
  int ret = Fun4AllReturnCodes::EVENT_OK;

  CentralityInfo *centInfo = findNode::getClass<CentralityInfo>(topNode, "CentralityInfo");
  if (centInfo)
  {
    m_data.cent = centInfo->get_centile(CentralityInfo::PROP::mbd_NS) * 100.0;
  }

  if (m_do_ep)
  {
    ret = process_event_plane(topNode);
  }

  if (ret != Fun4AllReturnCodes::EVENT_OK)
  {
    return ret;
  }

  if (m_do_sepd)
  {
    ret = process_sepd(topNode);
  }

  if (ret != Fun4AllReturnCodes::EVENT_OK)
  {
    return ret;
  }

  if (m_do_mbd)
  {
    ret = process_mbd(topNode);
  }

  if (ret != Fun4AllReturnCodes::EVENT_OK)
  {
    return ret;
  }

  if (m_do_hist && m_do_sepd && m_do_mbd)
  {
    double sepd_total = m_data.sepd_charge_south + m_data.sepd_charge_north;
    double mbd_total = m_data.mbd_charge_south + m_data.mbd_charge_north;
    if (m_hists.h2SEPD_MBD_Total_Charge)
    {
      m_hists.h2SEPD_MBD_Total_Charge->Fill(mbd_total, sepd_total);
    }
  }

  return ret;
}

//____________________________________________________________________________..
int GlobalQA::process_event_plane(PHCompositeNode *topNode)
{
  EventHeader *eventInfo = findNode::getClass<EventHeader>(topNode, "EventHeader");
  int event_id = eventInfo ? eventInfo->get_EvtSequence() : -1;

  // get event plane map
  EventplaneinfoMap *epmap = findNode::getClass<EventplaneinfoMap>(topNode, "EventplaneinfoMap");
  if (!epmap || epmap->empty())
  {
    std::cout << "Aborting Run: Event Plane Map null or empty" << std::endl;
    return Fun4AllReturnCodes::ABORTRUN;
  }

  Eventplaneinfo *epd_S = epmap->get(EventplaneinfoMap::sEPDS);
  Eventplaneinfo *epd_N = epmap->get(EventplaneinfoMap::sEPDN);
  Eventplaneinfo *epd_NS = epmap->get(EventplaneinfoMap::sEPDNS);

  // ensure the ptrs are valid
  if (!epd_S || !epd_N || !epd_NS)
  {
    std::cout << "Aborting Run: Event Plane map pointers invalid" << std::endl;
    return Fun4AllReturnCodes::ABORTRUN;
  }

  std::pair<double, double> Q_S_2_raw = epd_S->get_qvector_raw(2);
  std::pair<double, double> Q_N_2_raw = epd_N->get_qvector_raw(2);
  std::pair<double, double> Q_NS_2_raw = epd_NS->get_qvector_raw(2);

  std::pair<double, double> Q_S_2 = epd_S->get_qvector(2);
  std::pair<double, double> Q_N_2 = epd_N->get_qvector(2);
  std::pair<double, double> Q_NS_2 = epd_NS->get_qvector(2);

  double _2psi2_raw_S = 2*epd_S->GetPsi(Q_S_2_raw.first, Q_S_2_raw.second, 2);
  double _2psi2_raw_N = 2*epd_N->GetPsi(Q_N_2_raw.first, Q_N_2_raw.second, 2);
  double _2psi2_raw_NS = 2*epd_NS->GetPsi(Q_NS_2_raw.first, Q_NS_2_raw.second, 2);

  double _2psi2_S = 2*epd_S->GetPsi(Q_S_2.first, Q_S_2.second, 2);
  double _2psi2_N = 2*epd_N->GetPsi(Q_N_2.first, Q_N_2.second, 2);
  double _2psi2_NS = 2*epd_NS->GetPsi(Q_NS_2.first, Q_NS_2.second, 2);

  m_data.psi2_raw_S = _2psi2_raw_S;
  m_data.psi2_raw_N = _2psi2_raw_N;
  m_data.psi2_raw_NS = _2psi2_raw_NS;

  m_data.psi2_S = _2psi2_S;
  m_data.psi2_N = _2psi2_N;
  m_data.psi2_NS = _2psi2_NS;

  EPStatus status = get_ep_status(epd_S, epd_N, epd_NS);

  if (m_do_hist)
  {
    m_hists.hEP_Status->Fill(status);

    const bool valid_cent = has_valid_cent();
    const double cent = m_data.cent;

    // raw angles: all events with sEPD charge in both arms (zero charge gives Q = (0, 0)) and finite Q
    if (status != EP_ZERO_CHARGE && status != EP_NON_FINITE)
    {
      m_hists.h2Psi2_raw_SN->Fill(_2psi2_raw_S, _2psi2_raw_N);

      if (valid_cent)
      {
        m_hists.h2Psi2_S_raw->Fill(_2psi2_raw_S, cent);
        m_hists.h2Psi2_N_raw->Fill(_2psi2_raw_N, cent);
        m_hists.h2Psi2_NS_raw->Fill(_2psi2_raw_NS, cent);
      }
    }

    // calibrated angles: calibrated events only, otherwise the "calibrated" Q is the raw one.
    // Calibrated implies a valid centrality in [0, m_ep_calib_cent_max).
    if (status == EP_CALIBRATED && valid_cent)
    {
      // 2D South vs North correlation (projections give South/North 1D angles)
      m_hists.h2Psi2_SN->Fill(_2psi2_S, _2psi2_N);

      // Delta Psi2 wrapped to [-pi, pi]
      constexpr double pi = std::numbers::pi;
      double dpsi = _2psi2_S - _2psi2_N;
      while (dpsi > pi)
      {
        dpsi -= 2 * pi;
      }
      while (dpsi < -pi)
      {
        dpsi += 2 * pi;
      }
      m_hists.hDeltaPsi2_SN->Fill(dpsi);

      double cos_2dpsi = std::cos(dpsi);
      m_hists.hCos2DeltaPsi2->Fill(cos_2dpsi);

      m_hists.h2Psi2_S->Fill(_2psi2_S, cent);
      m_hists.h2Psi2_N->Fill(_2psi2_N, cent);
      m_hists.h2Psi2_NS->Fill(_2psi2_NS, cent);

      m_hists.pCos2DeltaPsi2_Cent->Fill(cent, cos_2dpsi);
      m_hists.pSin2DeltaPsi2_Cent->Fill(cent, std::sin(dpsi));

      // n = 3 sub-event resolution (periodic in 3 Psi_3, no wrapping needed)
      std::pair<double, double> Q_S_3 = epd_S->get_qvector(3);
      std::pair<double, double> Q_N_3 = epd_N->get_qvector(3);
      double psi3_S = epd_S->GetPsi(Q_S_3.first, Q_S_3.second, 3);
      double psi3_N = epd_N->GetPsi(Q_N_3.first, Q_N_3.second, 3);
      m_hists.pCos3DeltaPsi3_Cent->Fill(cent, std::cos(3 * (psi3_S - psi3_N)));

      // Q_2 moments per calibration stage and Fourier flatness of the calibrated Psi_2
      const std::array<const Eventplaneinfo *, kNEPDet> eps = {epd_S, epd_N, epd_NS};
      const std::array<double, kNEPDet> two_psi2 = {_2psi2_S, _2psi2_N, _2psi2_NS};

      for (unsigned int det = 0; det < kNEPDet; ++det)
      {
        const std::array<std::pair<double, double>, kNQStage> qs = {eps[det]->get_qvector_raw(2),
                                                                     eps[det]->get_qvector_recentered(2),
                                                                     eps[det]->get_qvector(2)};
        for (unsigned int stage = 0; stage < kNQStage; ++stage)
        {
          const auto &[qx, qy] = qs[stage];
          const std::array<double, kNQMoment> moments = {qx, qy, qx * qx, qy * qy, qx * qy};
          for (unsigned int mom = 0; mom < kNQMoment; ++mom)
          {
            m_hists.pQ2_Moment[stage][det][mom]->Fill(cent, moments[mom]);
          }
        }

        for (unsigned int k = 1; k <= kNFlatTerms; ++k)
        {
          m_hists.pFlat_Cos[det][k - 1]->Fill(cent, std::cos(k * two_psi2[det]));
          m_hists.pFlat_Sin[det][k - 1]->Fill(cent, std::sin(k * two_psi2[det]));
        }
      }
    }
  }

  if (Verbosity() > 0)
  {
    std::cout << "GlobalQA::process_event_plane - [Event " << event_id << "] "
              << "psi2_raw (S/N/NS): " << m_data.psi2_raw_S << " / " << m_data.psi2_raw_N << " / " << m_data.psi2_raw_NS
              << " | psi2 (S/N/NS): " << m_data.psi2_S << " / " << m_data.psi2_N << " / " << m_data.psi2_NS
              << std::endl;
  }
  if (Verbosity() > 1)
  {
    std::cout << "    Q2_raw: S=(" << Q_S_2_raw.first << ", " << Q_S_2_raw.second << ")"
              << " N=(" << Q_N_2_raw.first << ", " << Q_N_2_raw.second << ")"
              << " NS=(" << Q_NS_2_raw.first << ", " << Q_NS_2_raw.second << ")" << std::endl
              << "    Q2_calib: S=(" << Q_S_2.first << ", " << Q_S_2.second << ")"
              << " N=(" << Q_N_2.first << ", " << Q_N_2.second << ")"
              << " NS=(" << Q_NS_2.first << ", " << Q_NS_2.second << ")" << std::endl;
  }

  return Fun4AllReturnCodes::EVENT_OK;
}

//____________________________________________________________________________..
int GlobalQA::process_sepd(PHCompositeNode *topNode)
{
  EventHeader *eventInfo = findNode::getClass<EventHeader>(topNode, "EventHeader");
  int event_id = eventInfo ? eventInfo->get_EvtSequence() : -1;

  TowerInfoContainer *towerinfosEPD = findNode::getClass<TowerInfoContainer>(topNode, "TOWERINFO_CALIB_SEPD");
  if (!towerinfosEPD)
  {
    std::cout << "Aborting Run: TOWERINFO_CALIB_SEPD node missing (is EpdReco registered?)" << std::endl;
    return Fun4AllReturnCodes::ABORTRUN;
  }

  if (m_do_hist)
  {
    m_hists.hSEPD_NEvents->Fill(0.5);
  }

  // sepd
  unsigned int nchannels_epd = towerinfosEPD->size();

  double sepd_total_charge_south = 0;
  double sepd_total_charge_north = 0;
  unsigned int nhits_south = 0;
  unsigned int nhits_north = 0;

  for (unsigned int channel = 0; channel < nchannels_epd; ++channel)
  {
    unsigned int key = TowerInfoDefs::encode_epd(channel);

    TowerInfo *tower = towerinfosEPD->get_tower_at_channel(channel);
    if (!tower)
    {
      continue;
    }

    int rbin = TowerInfoDefs::get_epd_rbin(key);

    // Skip Innermost Ring
    if (m_skipRing0 && rbin == 0)
    {
      continue;
    }

    double charge = tower->get_energy();

    // skip non-finite charge (NaN passes every comparison below, inf would dominate the sums)
    if (!std::isfinite(charge))
    {
      if (m_do_hist)
      {
        m_hists.hSEPD_Channel_NonFinite->Fill(channel);
      }
      if (Verbosity() > 0)
      {
        std::cout << "GlobalQA::process_sepd - [Event " << event_id << "] non-finite charge " << charge
                  << " in channel " << channel << ", skipped" << std::endl;
      }
      continue;
    }

    unsigned int arm = TowerInfoDefs::get_epd_arm(key);

    // skip charge below minimum threshold
    if (charge < m_sepd_channel_threshold)
    {
      continue;
    }

    if (m_do_hist)
    {
      if (m_hists.h2SEPD_Channel_Charge)
      {
        m_hists.h2SEPD_Channel_Charge->Fill(channel, charge);
      }
      if (m_hists.pSEPD_Channel_Charge)
      {
        m_hists.pSEPD_Channel_Charge->Fill(channel, charge);
      }
      if (arm < kNArm)
      {
        int phibin = TowerInfoDefs::get_epd_phibin(key);
        m_hists.h2SEPD_RingPhi_Hits[arm]->Fill(rbin, phibin);
        m_hists.pSEPD_RingPhi_Charge[arm]->Fill(rbin, phibin, charge);
      }
      if (arm == 0)
      {
        if (m_hists.pSEPD_Ring_Charge_South)
        {
          m_hists.pSEPD_Ring_Charge_South->Fill(rbin, charge);
        }
      }
      else
      {
        if (m_hists.pSEPD_Ring_Charge_North)
        {
          m_hists.pSEPD_Ring_Charge_North->Fill(rbin, charge);
        }
      }
    }

    if (arm == 0)
    {
      sepd_total_charge_south += charge;
      ++nhits_south;
    }
    else
    {
      sepd_total_charge_north += charge;
      ++nhits_north;
    }
  }

  m_data.sepd_charge_south = sepd_total_charge_south;
  m_data.sepd_charge_north = sepd_total_charge_north;
  double sepd_total_charge = sepd_total_charge_south + sepd_total_charge_north;
  unsigned int nhits_total = nhits_south + nhits_north;

  if (m_do_hist)
  {
    if (m_hists.hSEPD_Charge_Total)
    {
      m_hists.hSEPD_Charge_Total->Fill(sepd_total_charge);
    }
    if (m_hists.hSEPD_Hits_Total)
    {
      m_hists.hSEPD_Hits_Total->Fill(nhits_total);
    }

    if (m_hists.h2SEPD_Charge_South_North)
    {
      m_hists.h2SEPD_Charge_South_North->Fill(sepd_total_charge_south, sepd_total_charge_north);
    }
    if (m_hists.h2SEPD_Hits_South_North)
    {
      m_hists.h2SEPD_Hits_South_North->Fill(nhits_south, nhits_north);
    }
    if (has_valid_cent())
    {
      m_hists.h2SEPD_Charge_Centrality->Fill(m_data.cent, sepd_total_charge);
    }
  }

  if (Verbosity() > 0)
  {
    std::cout << "GlobalQA::process_sepd - [Event " << event_id << "] "
              << "Charge S: " << m_data.sepd_charge_south << " (hits: " << nhits_south << ")"
              << " | Charge N: " << m_data.sepd_charge_north << " (hits: " << nhits_north << ")"
              << " | Total: " << sepd_total_charge
              << " (threshold: " << m_sepd_channel_threshold << ")"
              << std::endl;
  }

  return Fun4AllReturnCodes::EVENT_OK;
}

//____________________________________________________________________________..
int GlobalQA::process_mbd(PHCompositeNode *topNode)
{
  EventHeader *eventInfo = findNode::getClass<EventHeader>(topNode, "EventHeader");
  int event_id = eventInfo ? eventInfo->get_EvtSequence() : -1;

  PdbParameterMap *pdb = findNode::getClass<PdbParameterMap>(topNode, "MinBiasParams");
  if (!pdb)
  {
    std::cout << "Aborting Run: MinBiasParams node missing" << std::endl;
    return Fun4AllReturnCodes::ABORTRUN;
  }

  PHParameters pdb_params("MinBiasParams");
  pdb_params.FillFrom(pdb);

  double mbd_total_charge_south = pdb_params.get_double_param("minbias_mbd_total_charge_south");
  double mbd_total_charge_north = pdb_params.get_double_param("minbias_mbd_total_charge_north");

  m_data.mbd_charge_south = mbd_total_charge_south;
  m_data.mbd_charge_north = mbd_total_charge_north;
  double mbd_total_charge = mbd_total_charge_south + mbd_total_charge_north;

  if (m_do_hist)
  {
    if (m_hists.hMBD_Charge_Total)
    {
      m_hists.hMBD_Charge_Total->Fill(mbd_total_charge);
    }

    if (m_hists.h2MBD_Charge_South_North)
    {
      m_hists.h2MBD_Charge_South_North->Fill(mbd_total_charge_south, mbd_total_charge_north);
    }
  }

  if (Verbosity() > 0)
  {
    std::cout << "GlobalQA::process_mbd - [Event " << event_id << "] "
              << "Charge S: " << m_data.mbd_charge_south
              << " | Charge N: " << m_data.mbd_charge_north
              << " | Total: " << mbd_total_charge
              << std::endl;
  }

  return Fun4AllReturnCodes::EVENT_OK;
}

int GlobalQA::ResetEvent([[maybe_unused]] PHCompositeNode *topNode)
{
  // sEPD - Event Plane
  m_data.psi2_raw_S = 0;
  m_data.psi2_raw_N = 0;
  m_data.psi2_raw_NS = 0;

  m_data.psi2_S = 0;
  m_data.psi2_N = 0;
  m_data.psi2_NS = 0;

  // sEPD - QA
  m_data.sepd_charge_south = 0;
  m_data.sepd_charge_north = 0;

  // MBD - QA
  m_data.mbd_charge_south = 0;
  m_data.mbd_charge_north = 0;

  m_data.cent = std::numeric_limits<double>::quiet_NaN();

  return Fun4AllReturnCodes::EVENT_OK;
}

//____________________________________________________________________________..
bool GlobalQA::has_valid_cent() const
{
  return std::isfinite(m_data.cent) && m_data.cent >= 0.0 && m_data.cent <= 100.0;
}

//____________________________________________________________________________..
GlobalQA::EPStatus GlobalQA::get_ep_status(const Eventplaneinfo *epd_S, const Eventplaneinfo *epd_N, const Eventplaneinfo *epd_NS) const
{
  // EventPlaneReco resets all Q vectors to exactly (0, 0) each event and only sets the
  // recentered ones when it calibrates; zero charge in an arm leaves the raw ones at (0, 0)
  auto is_zero = [](const std::pair<double, double> &q)
  { return q.first == 0.0 && q.second == 0.0; };
  auto is_finite = [](const std::pair<double, double> &q)
  { return std::isfinite(q.first) && std::isfinite(q.second); };

  // every Q vector the histograms use (raw, recentered, calibrated; n = 2 and 3)
  for (const Eventplaneinfo *ep : {epd_S, epd_N, epd_NS})
  {
    for (unsigned int n : {2U, 3U})
    {
      if (!is_finite(ep->get_qvector_raw(n)) || !is_finite(ep->get_qvector_recentered(n)) || !is_finite(ep->get_qvector(n)))
      {
        return EP_NON_FINITE;
      }
    }
  }

  if (is_zero(epd_S->get_qvector_raw(2)) || is_zero(epd_N->get_qvector_raw(2)))
  {
    return EP_ZERO_CHARGE;
  }

  if (is_zero(epd_S->get_qvector_recentered(2)) && is_zero(epd_N->get_qvector_recentered(2)))
  {
    // same centrality definition and range as EventPlaneReco
    if (!std::isfinite(m_data.cent) || m_data.cent < 0.0 || m_data.cent >= m_ep_calib_cent_max)
    {
      return EP_CENT_OUT_RANGE;
    }
    return EP_NO_CALIB;
  }

  return EP_CALIBRATED;
}

int GlobalQA::End([[maybe_unused]] PHCompositeNode *topNode)
{
  std::cout << "GlobalQA::End" << std::endl;

  return Fun4AllReturnCodes::EVENT_OK;
}
