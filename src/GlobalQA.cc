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
#include <TTree.h>

#include <treefiller/TreeFiller.h>

#include <cmath>
#include <format>
#include <iostream>
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

      m_hists.pSEPD_Ring_Charge_South = new TProfile("pSEPD_Ring_Charge_South", "sEPD South Mean Charge per Ring; Radial Bin; #LTCharge#GT [MIP]",
                                                     m_hist_config.bins_sepd_rings, m_hist_config.sepd_ring_low, m_hist_config.sepd_ring_high);
      se->registerHisto(m_hists.pSEPD_Ring_Charge_South);

      m_hists.pSEPD_Ring_Charge_North = new TProfile("pSEPD_Ring_Charge_North", "sEPD North Mean Charge per Ring; Radial Bin; #LTCharge#GT [MIP]",
                                                     m_hist_config.bins_sepd_rings, m_hist_config.sepd_ring_low, m_hist_config.sepd_ring_high);
      se->registerHisto(m_hists.pSEPD_Ring_Charge_North);
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
    }
  }

  return Fun4AllReturnCodes::EVENT_OK;
}

//____________________________________________________________________________..
int GlobalQA::process_event(PHCompositeNode *topNode)
{
  int ret = Fun4AllReturnCodes::EVENT_OK;

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

  if (m_do_hist)
  {
    // 2D South vs North correlations (projections give South/North 1D angles)
    if (m_hists.h2Psi2_raw_SN)
    {
      m_hists.h2Psi2_raw_SN->Fill(_2psi2_raw_S, _2psi2_raw_N);
    }
    if (m_hists.h2Psi2_SN)
    {
      m_hists.h2Psi2_SN->Fill(_2psi2_S, _2psi2_N);
    }

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
    if (m_hists.hDeltaPsi2_SN)
    {
      m_hists.hDeltaPsi2_SN->Fill(dpsi);
    }

    double cos_2dpsi = std::cos(dpsi);
    if (m_hists.hCos2DeltaPsi2)
    {
      m_hists.hCos2DeltaPsi2->Fill(cos_2dpsi);
    }

    CentralityInfo *centInfo = findNode::getClass<CentralityInfo>(topNode, "CentralityInfo");
    if (centInfo)
    {
      double cent = centInfo->get_centile(CentralityInfo::PROP::mbd_NS) * 100.0;
      if (std::isfinite(cent) && cent >= 0.0 && cent <= 100.0)
      {
        if (m_hists.h2Psi2_S_raw)
        {
          m_hists.h2Psi2_S_raw->Fill(_2psi2_raw_S, cent);
        }
        if (m_hists.h2Psi2_N_raw)
        {
          m_hists.h2Psi2_N_raw->Fill(_2psi2_raw_N, cent);
        }
        if (m_hists.h2Psi2_NS_raw)
        {
          m_hists.h2Psi2_NS_raw->Fill(_2psi2_raw_NS, cent);
        }

        if (m_hists.h2Psi2_S)
        {
          m_hists.h2Psi2_S->Fill(_2psi2_S, cent);
        }
        if (m_hists.h2Psi2_N)
        {
          m_hists.h2Psi2_N->Fill(_2psi2_N, cent);
        }
        if (m_hists.h2Psi2_NS)
        {
          m_hists.h2Psi2_NS->Fill(_2psi2_NS, cent);
        }

        if (m_hists.pCos2DeltaPsi2_Cent)
        {
          m_hists.pCos2DeltaPsi2_Cent->Fill(cent, cos_2dpsi);
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
    return Fun4AllReturnCodes::ABORTRUN;
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

  return Fun4AllReturnCodes::EVENT_OK;
}

int GlobalQA::End([[maybe_unused]] PHCompositeNode *topNode)
{
  std::cout << "GlobalQA::End" << std::endl;

  return Fun4AllReturnCodes::EVENT_OK;
}
