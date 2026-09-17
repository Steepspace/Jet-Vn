#include "EventQA.h"

#include <treefiller/TreeFiller.h>

#include <format>
#include <iostream>

#include <ffaobjects/EventHeader.h>
#include <fun4all/Fun4AllReturnCodes.h>
#include <fun4all/Fun4AllServer.h>
#include <phool/PHCompositeNode.h>
#include <phool/getClass.h>

#include <pdbcalbase/PdbParameterMap.h>
#include <phparameter/PHParameters.h>

#include <globalvertex/GlobalVertex.h>
#include <globalvertex/GlobalVertexMap.h>

#include <calotrigger/MinimumBiasClassifier.h>
#include <calotrigger/MinimumBiasInfo.h>
#include <centrality/CentralityInfo.h>

#include <calotrigger/TriggerAnalyzer.h>

#include <TH1F.h>
#include <TH2F.h>
#include <TTree.h>

EventQA::EventQA(const std::string &name)
  : SubsysReco(name),
    hZVertexTrig(TrigIdx::NUM_TRIG),
    hCentralityTrig(TrigIdx::NUM_TRIG),
    hCentralityZ150Trig(TrigIdx::NUM_TRIG),
    hCentralityZOuterTrig(TrigIdx::NUM_TRIG),
    h2ZVertexCentralityTrig(TrigIdx::NUM_TRIG)
{
}

int EventQA::Init([[maybe_unused]] PHCompositeNode *topNode)
{
  Fun4AllServer *se = Fun4AllServer::instance();

  m_triggerAnalyzer = std::make_unique<TriggerAnalyzer>();

  if (m_do_hist)
  {
    hEvent = new TH1F("hEvent", "Event Type; Type; Events", static_cast<unsigned int>(m_eventType.size()), 0, static_cast<double>(m_eventType.size()));
    se->registerHisto(hEvent);

    hEventMinBias = new TH1F("hEventMinBias", "Event Type; Type; Events", static_cast<unsigned int>(m_MinBias_Type.size()), 0, static_cast<double>(m_MinBias_Type.size()));
    se->registerHisto(hEventMinBias);

    // Event Trigger Counter Histogram
    std::vector<std::string> eventTypeTrigger{"|z| < 10 cm"};
    for (const auto &trig : m_triggernames)
    {
      eventTypeTrigger.push_back(trig);
    }

    hEventTrigger = new TH1F("hEventTrigger", "Event Selection; Type; Events", static_cast<unsigned int>(eventTypeTrigger.size()), 0, static_cast<double>(eventTypeTrigger.size()));
    for (unsigned int i = 0; i < eventTypeTrigger.size(); ++i)
    {
      hEventTrigger->GetXaxis()->SetBinLabel(i + 1, eventTypeTrigger[i].c_str());
    }
    se->registerHisto(hEventTrigger);

    hVtxZ = new TH1F("hVtxZ", "Z Vertex; z [cm]; Events", m_hist_config.m_bins_zvtx, m_hist_config.m_zvtx_low, m_hist_config.m_zvtx_high);
    se->registerHisto(hVtxZ);

    hVtxZ_MB = new TH1F("hVtxZ_MB", "Z Vertex; z [cm]; Events", m_hist_config.m_bins_zvtx, m_hist_config.m_zvtx_low, m_hist_config.m_zvtx_high);
    se->registerHisto(hVtxZ_MB);

    hZVertex = new TH1F("hZVertex", "Min Bias; Z [cm]; Events", m_hist_config.m_bins_zvtx, m_hist_config.m_zvtx_low, m_hist_config.m_zvtx_high);
    se->registerHisto(hZVertex);

    // Centrality Histograms
    hCentrality = new TH1F("hCentrality", "|z| < 10 cm and MB; Centrality [%]; Events", m_hist_config.m_bins_cent, m_hist_config.m_cent_low, m_hist_config.m_cent_high);
    se->registerHisto(hCentrality);

    hCentralityZ150 = new TH1F("hCentralityZ150", "|z| < 150 cm and MB; Centrality [%]; Events", m_hist_config.m_bins_cent, m_hist_config.m_cent_low, m_hist_config.m_cent_high);
    se->registerHisto(hCentralityZ150);

    hCentralityZOuter = new TH1F("hCentralityZOuter", "10 cm < |z| < 150 cm and MB; Centrality [%]; Events", m_hist_config.m_bins_cent, m_hist_config.m_cent_low, m_hist_config.m_cent_high);
    se->registerHisto(hCentralityZOuter);

    // 2D Vertex vs Centrality
    h2ZVertexCentrality = new TH2F("h2ZVertexCentrality", "Min Bias; Z [cm]; Centrality [%]", m_hist_config.m_bins_zvtx, m_hist_config.m_zvtx_low, m_hist_config.m_zvtx_high, m_hist_config.m_bins_cent, m_hist_config.m_cent_low, m_hist_config.m_cent_high);
    se->registerHisto(h2ZVertexCentrality);

    for (size_t i = 0; i < m_triggerBits.size(); ++i)
    {
      int triggerIdx = m_triggerBits[i];
      const auto &trig = m_triggernames[i];

      size_t idx_relaxed = 2 * i;
      size_t idx_tight = 2 * i + 1;

      // Relaxed (trigger-only, without offline MB)
      std::string title_centrality = std::format("|z| < 10 cm and {}; Centrality [%]; Events", trig);
      std::string title_centralityZ150 = std::format("|z| < 150 cm and {}; Centrality [%]; Events", trig);
      std::string title_centralityZOuter = std::format("10 cm < |z| < 150 cm and {}; Centrality [%]; Events", trig);

      std::string name_centrality = std::format("hCentrality_Trig{}", triggerIdx);
      std::string name_centralityZ150 = std::format("hCentralityZ150_Trig{}", triggerIdx);
      std::string name_centralityZOuter = std::format("hCentralityZOuter_Trig{}", triggerIdx);

      hCentralityTrig[idx_relaxed] = new TH1F(name_centrality.c_str(), title_centrality.c_str(), m_hist_config.m_bins_cent, m_hist_config.m_cent_low, m_hist_config.m_cent_high);
      hCentralityZ150Trig[idx_relaxed] = new TH1F(name_centralityZ150.c_str(), title_centralityZ150.c_str(), m_hist_config.m_bins_cent, m_hist_config.m_cent_low, m_hist_config.m_cent_high);
      hCentralityZOuterTrig[idx_relaxed] = new TH1F(name_centralityZOuter.c_str(), title_centralityZOuter.c_str(), m_hist_config.m_bins_cent, m_hist_config.m_cent_low, m_hist_config.m_cent_high);

      se->registerHisto(hCentralityTrig[idx_relaxed]);
      se->registerHisto(hCentralityZ150Trig[idx_relaxed]);
      se->registerHisto(hCentralityZOuterTrig[idx_relaxed]);

      std::string title_h1 = std::format("{}; Z [cm]; Events", trig);
      std::string title_h2 = std::format("{}; Z [cm]; Centrality [%]", trig);

      std::string name_h1 = std::format("hZVertex_Trig{}", triggerIdx);
      std::string name_h2 = std::format("h2ZVertexCentrality_Trig{}", triggerIdx);

      hZVertexTrig[idx_relaxed] = new TH1F(name_h1.c_str(), title_h1.c_str(), m_hist_config.m_bins_zvtx, m_hist_config.m_zvtx_low, m_hist_config.m_zvtx_high);
      h2ZVertexCentralityTrig[idx_relaxed] = new TH2F(name_h2.c_str(), title_h2.c_str(), m_hist_config.m_bins_zvtx, m_hist_config.m_zvtx_low, m_hist_config.m_zvtx_high, m_hist_config.m_bins_cent, m_hist_config.m_cent_low, m_hist_config.m_cent_high);

      se->registerHisto(hZVertexTrig[idx_relaxed]);
      se->registerHisto(h2ZVertexCentralityTrig[idx_relaxed]);

      // Tight (trigger + offline MB)
      std::string title_centrality_MB = std::format("|z| < 10 cm and {} and MB; Centrality [%]; Events", trig);
      std::string title_centralityZ150_MB = std::format("|z| < 150 cm and {} and MB; Centrality [%]; Events", trig);
      std::string title_centralityZOuter_MB = std::format("10 cm < |z| < 150 cm and {} and MB; Centrality [%]; Events", trig);

      std::string name_centrality_MB = std::format("hCentrality_Trig{}_MB", triggerIdx);
      std::string name_centralityZ150_MB = std::format("hCentralityZ150_Trig{}_MB", triggerIdx);
      std::string name_centralityZOuter_MB = std::format("hCentralityZOuter_Trig{}_MB", triggerIdx);

      hCentralityTrig[idx_tight] = new TH1F(name_centrality_MB.c_str(), title_centrality_MB.c_str(), m_hist_config.m_bins_cent, m_hist_config.m_cent_low, m_hist_config.m_cent_high);
      hCentralityZ150Trig[idx_tight] = new TH1F(name_centralityZ150_MB.c_str(), title_centralityZ150_MB.c_str(), m_hist_config.m_bins_cent, m_hist_config.m_cent_low, m_hist_config.m_cent_high);
      hCentralityZOuterTrig[idx_tight] = new TH1F(name_centralityZOuter_MB.c_str(), title_centralityZOuter_MB.c_str(), m_hist_config.m_bins_cent, m_hist_config.m_cent_low, m_hist_config.m_cent_high);

      se->registerHisto(hCentralityTrig[idx_tight]);
      se->registerHisto(hCentralityZ150Trig[idx_tight]);
      se->registerHisto(hCentralityZOuterTrig[idx_tight]);

      std::string title_h1_MB = std::format("{} and MB; Z [cm]; Events", trig);
      std::string title_h2_MB = std::format("{} and MB; Z [cm]; Centrality [%]", trig);

      std::string name_h1_MB = std::format("hZVertex_Trig{}_MB", triggerIdx);
      std::string name_h2_MB = std::format("h2ZVertexCentrality_Trig{}_MB", triggerIdx);

      hZVertexTrig[idx_tight] = new TH1F(name_h1_MB.c_str(), title_h1_MB.c_str(), m_hist_config.m_bins_zvtx, m_hist_config.m_zvtx_low, m_hist_config.m_zvtx_high);
      h2ZVertexCentralityTrig[idx_tight] = new TH2F(name_h2_MB.c_str(), title_h2_MB.c_str(), m_hist_config.m_bins_zvtx, m_hist_config.m_zvtx_low, m_hist_config.m_zvtx_high, m_hist_config.m_bins_cent, m_hist_config.m_cent_low, m_hist_config.m_cent_high);

      se->registerHisto(hZVertexTrig[idx_tight]);
      se->registerHisto(h2ZVertexCentralityTrig[idx_tight]);
    }

    for (unsigned int i = 0; i < m_eventType.size(); ++i)
    {
      hEvent->GetXaxis()->SetBinLabel(i + 1, m_eventType[i].c_str());
    }

    for (unsigned int i = 0; i < m_MinBias_Type.size(); ++i)
    {
      hEventMinBias->GetXaxis()->SetBinLabel(i + 1, m_MinBias_Type[i].c_str());
    }
  }

  if (m_do_tree)
  {
    TTree *tree = TreeFiller::getTree();
    if (tree)
    {
      tree->Branch("run", &m_data.run);
      tree->Branch("event", &m_data.event);
      tree->Branch("zvtx", &m_data.zvtx);
      tree->Branch("centrality", &m_data.centrality);
    }
  }

  return Fun4AllReturnCodes::EVENT_OK;
}

int EventQA::process_event_check(PHCompositeNode *topNode)
{
  if (m_do_hist)
  {
    hEvent->Fill(static_cast<std::uint8_t>(EventType::ALL));
  }

  EventHeader *eventInfo = findNode::getClass<EventHeader>(topNode, "EventHeader");
  if (!eventInfo)
  {
    std::cout << "Aborting Run: EventHeader null" << std::endl;
    return Fun4AllReturnCodes::ABORTRUN;
  }

  m_data.run = eventInfo->get_RunNumber();
  m_data.event = eventInfo->get_EvtSequence();

  // zvertex
  double zvtx = -9999;
  GlobalVertexMap *vertexmap = findNode::getClass<GlobalVertexMap>(topNode, "GlobalVertexMap");

  if (!vertexmap)
  {
    std::cout << "Aborting Run: GlobalVertexMap null" << std::endl;
    return Fun4AllReturnCodes::ABORTRUN;
  }

  if (!vertexmap->empty())
  {
    GlobalVertex *vtx = vertexmap->begin()->second;
    if (vtx)
    {
      m_data.zvtx = vtx->get_z();
      zvtx = m_data.zvtx;

      if (m_do_hist)
      {
        hEvent->Fill(static_cast<std::uint8_t>(EventType::ZVTX));
      }
    }
  }

  if (m_do_hist)
  {
    hVtxZ->Fill(zvtx);
  }

  bool pass_zvtx10 = std::abs(zvtx) < m_cuts.m_zvtx_max;
  m_pass_Zvtx = pass_zvtx10;

  if (std::abs(zvtx) < m_cuts.m_zvtx_max_v2)
  {
    if (m_do_hist)
    {
      hEvent->Fill(static_cast<std::uint8_t>(EventType::ZVTX150));
      if (pass_zvtx10)
      {
        hEvent->Fill(static_cast<std::uint8_t>(EventType::ZVTX10));
        if (hEventTrigger)
        {
          hEventTrigger->Fill(0);
        }
      }
    }
  }

  // MBD Trigger
  m_triggerAnalyzer->decodeTriggers(topNode);

  m_didTrig14Fire = m_triggerAnalyzer->didTriggerFire(m_trig_14);
  m_didTrig12Fire = m_triggerAnalyzer->didTriggerFire(m_trig_12);

  bool mbd_trigger_fire = m_didTrig12Fire || m_didTrig14Fire;

  if (m_do_hist)
  {
    if (pass_zvtx10 && hEventTrigger)
    {
      if (m_didTrig12Fire)
      {
        hEventTrigger->Fill(1);
      }
      if (m_didTrig14Fire)
      {
        hEventTrigger->Fill(2);
      }
    }

    if (!vertexmap->empty())
    {
      if (m_didTrig12Fire && hZVertexTrig[TrigIdx::TRIG12])
      {
        hZVertexTrig[TrigIdx::TRIG12]->Fill(zvtx);
      }
      if (m_didTrig14Fire && hZVertexTrig[TrigIdx::TRIG14])
      {
        hZVertexTrig[TrigIdx::TRIG14]->Fill(zvtx);
      }
    }
  }

  if (pass_zvtx10 && mbd_trigger_fire)
  {
    if (m_do_hist)
    {
      hEvent->Fill(static_cast<std::uint8_t>(EventType::MB_TRIG));
    }
  }

  // Minimum Bias Classifier
  MinimumBiasInfo *m_mb_info = findNode::getClass<MinimumBiasInfo>(topNode, "MinimumBiasInfo");
  if (!m_mb_info)
  {
    std::cout << "Aborting Run: MinimumBiasInfo null" << std::endl;
    return Fun4AllReturnCodes::ABORTRUN;
  }

  // Minimum Bias Check
  PdbParameterMap *pdb = findNode::getClass<PdbParameterMap>(topNode, "MinBiasParams");
  if (!pdb)
  {
    std::cout << "Aborting Run: PdbParameterMap null" << std::endl;
    return Fun4AllReturnCodes::ABORTRUN;
  }

  PHParameters pdb_params("MinBiasParams");
  pdb_params.FillFrom(pdb);

  bool minbias_bkg_high = pdb_params.get_int_param("minbias_background_cut_fail");
  bool minbias_side_hit_low = pdb_params.get_int_param("minbias_two_hit_min_fail");
  bool minbias_zdc_low = pdb_params.get_int_param("minbias_zdc_energy_min_fail");
  bool minbias_mbd_high = pdb_params.get_int_param("minbias_mbd_total_energy_max_fail");

  if (Verbosity() > 0)
  {
    std::cout << "EventQA::process_event_check - [Event " << m_data.event << "] Run: " << m_data.run
              << " | zvtx: " << zvtx << " cm"
              << " | MBD Trig: " << mbd_trigger_fire << " (trig12=" << m_didTrig12Fire << ", trig14=" << m_didTrig14Fire << ")"
              << " | isAuAuMB: " << m_mb_info->isAuAuMinimumBias()
              << std::endl;
  }
  if (Verbosity() > 1)
  {
    std::cout << "    MinBias fails -> bkg_high: " << minbias_bkg_high
              << " | side_hit_low: " << minbias_side_hit_low
              << " | zdc_low: " << minbias_zdc_low
              << " | mbd_high: " << minbias_mbd_high
              << std::endl;
  }

  if (m_do_hist && pass_zvtx10 && mbd_trigger_fire)
  {
    if (minbias_bkg_high)
    {
      hEventMinBias->Fill(static_cast<std::uint8_t>(MinBiasType::BKG_HIGH));
    }
    if (minbias_side_hit_low)
    {
      hEventMinBias->Fill(static_cast<std::uint8_t>(MinBiasType::SIDE_HIT_LOW));
    }
    if (minbias_zdc_low)
    {
      hEventMinBias->Fill(static_cast<std::uint8_t>(MinBiasType::ZDC_LOW));
    }
    if (minbias_mbd_high)
    {
      hEventMinBias->Fill(static_cast<std::uint8_t>(MinBiasType::MBD_HIGH));
    }
  }

  // skip event if not fire MBD Trigger
  if (!mbd_trigger_fire)
  {
    if (Verbosity() > 0)
    {
      std::cout << "EventQA::process_event_check - [Event " << m_data.event << "] REJECTED: MBD trigger did not fire" << std::endl;
    }
    ++m_ctr["process_eventCheck_mbd_trigger_fail"];
    return (m_doAbort) ? Fun4AllReturnCodes::ABORTEVENT : Fun4AllReturnCodes::EVENT_OK;
  }

  // Check minimum bias
  if (!m_mb_info->isAuAuMinimumBias())
  {
    if (Verbosity() > 0)
    {
      std::cout << "EventQA::process_event_check - [Event " << m_data.event << "] isAuAuMinimumBias failed" << std::endl;
    }
    ++m_ctr["process_eventCheck_isAuAuMinBias_fail"];
  }
  else
  {
    m_pass_MB = true;

    if (m_do_hist)
    {
      hVtxZ_MB->Fill(zvtx);
      if (hZVertex)
      {
        hZVertex->Fill(zvtx);
      }
      if (m_didTrig12Fire && hZVertexTrig[TrigIdx::TRIG12_MB])
      {
        hZVertexTrig[TrigIdx::TRIG12_MB]->Fill(zvtx);
      }
      if (m_didTrig14Fire && hZVertexTrig[TrigIdx::TRIG14_MB])
      {
        hZVertexTrig[TrigIdx::TRIG14_MB]->Fill(zvtx);
      }
    }
  }

  if (Verbosity() > 0)
  {
    std::cout << "EventQA::process_event_check - [Event " << m_data.event << "] PASSED event selection" << std::endl;
  }

  return Fun4AllReturnCodes::EVENT_OK;
}

int EventQA::process_centrality(PHCompositeNode *topNode)
{
  CentralityInfo *centInfo = findNode::getClass<CentralityInfo>(topNode, "CentralityInfo");
  if (!centInfo)
  {
    std::cout << "Aborting Run: CentralityInfo null" << std::endl;
    return Fun4AllReturnCodes::ABORTRUN;
  }

  m_data.centrality = centInfo->get_centile(CentralityInfo::PROP::mbd_NS) * 100;
  double cent = m_data.centrality;

  if (Verbosity() > 0)
  {
    std::cout << "EventQA::process_centrality - [Event " << m_data.event << "] Centrality: " << cent << "% (cut < " << m_cuts.m_cent_max << "%)" << std::endl;
  }

  if (!std::isfinite(cent) || cent < 0 || cent >= m_hist_config.m_cent_high)
  {
    if (Verbosity() > 0)
    {
      std::cout << std::format("EventQA::process_centrality - [Event {}] Invalid centrality centile ({:.2f}). Expected [0, {}).",
                               m_data.event, cent, m_hist_config.m_cent_high) << std::endl;
    }
    ++m_ctr["events_centrality_bad"];
    return (m_doAbort) ? Fun4AllReturnCodes::ABORTEVENT : Fun4AllReturnCodes::EVENT_OK;
  }

  if (m_do_hist)
  {
    // Histograms requiring offline MB
    if (m_pass_MB)
    {
      h2ZVertexCentrality->Fill(m_data.zvtx, cent);

      if (std::abs(m_data.zvtx) < m_cuts.m_zvtx_max_v2)
      {
        hCentralityZ150->Fill(cent);

        if (m_pass_Zvtx)
        {
          hCentrality->Fill(cent);
        }
        else
        {
          hCentralityZOuter->Fill(cent);
        }
      }
    }

    // Trigger 12
    if (m_didTrig12Fire)
    {
      // Relaxed: trigger only
      if (h2ZVertexCentralityTrig[TrigIdx::TRIG12])
      {
        h2ZVertexCentralityTrig[TrigIdx::TRIG12]->Fill(m_data.zvtx, cent);
      }
      if (std::abs(m_data.zvtx) < m_cuts.m_zvtx_max_v2)
      {
        if (hCentralityZ150Trig[TrigIdx::TRIG12])
        {
          hCentralityZ150Trig[TrigIdx::TRIG12]->Fill(cent);
        }
        if (m_pass_Zvtx)
        {
          if (hCentralityTrig[TrigIdx::TRIG12])
          {
            hCentralityTrig[TrigIdx::TRIG12]->Fill(cent);
          }
        }
        else
        {
          if (hCentralityZOuterTrig[TrigIdx::TRIG12])
          {
            hCentralityZOuterTrig[TrigIdx::TRIG12]->Fill(cent);
          }
        }
      }

      // Tight: trigger + offline MB
      if (m_pass_MB)
      {
        if (h2ZVertexCentralityTrig[TrigIdx::TRIG12_MB])
        {
          h2ZVertexCentralityTrig[TrigIdx::TRIG12_MB]->Fill(m_data.zvtx, cent);
        }
        if (std::abs(m_data.zvtx) < m_cuts.m_zvtx_max_v2)
        {
          if (hCentralityZ150Trig[TrigIdx::TRIG12_MB])
          {
            hCentralityZ150Trig[TrigIdx::TRIG12_MB]->Fill(cent);
          }
          if (m_pass_Zvtx)
          {
            if (hCentralityTrig[TrigIdx::TRIG12_MB])
            {
              hCentralityTrig[TrigIdx::TRIG12_MB]->Fill(cent);
            }
          }
          else
          {
            if (hCentralityZOuterTrig[TrigIdx::TRIG12_MB])
            {
              hCentralityZOuterTrig[TrigIdx::TRIG12_MB]->Fill(cent);
            }
          }
        }
      }
    }

    // Trigger 14
    if (m_didTrig14Fire)
    {
      // Relaxed: trigger only
      if (h2ZVertexCentralityTrig[TrigIdx::TRIG14])
      {
        h2ZVertexCentralityTrig[TrigIdx::TRIG14]->Fill(m_data.zvtx, cent);
      }
      if (std::abs(m_data.zvtx) < m_cuts.m_zvtx_max_v2)
      {
        if (hCentralityZ150Trig[TrigIdx::TRIG14])
        {
          hCentralityZ150Trig[TrigIdx::TRIG14]->Fill(cent);
        }
        if (m_pass_Zvtx)
        {
          if (hCentralityTrig[TrigIdx::TRIG14])
          {
            hCentralityTrig[TrigIdx::TRIG14]->Fill(cent);
          }
        }
        else
        {
          if (hCentralityZOuterTrig[TrigIdx::TRIG14])
          {
            hCentralityZOuterTrig[TrigIdx::TRIG14]->Fill(cent);
          }
        }
      }

      // Tight: trigger + offline MB
      if (m_pass_MB)
      {
        if (h2ZVertexCentralityTrig[TrigIdx::TRIG14_MB])
        {
          h2ZVertexCentralityTrig[TrigIdx::TRIG14_MB]->Fill(m_data.zvtx, cent);
        }
        if (std::abs(m_data.zvtx) < m_cuts.m_zvtx_max_v2)
        {
          if (hCentralityZ150Trig[TrigIdx::TRIG14_MB])
          {
            hCentralityZ150Trig[TrigIdx::TRIG14_MB]->Fill(cent);
          }
          if (m_pass_Zvtx)
          {
            if (hCentralityTrig[TrigIdx::TRIG14_MB])
            {
              hCentralityTrig[TrigIdx::TRIG14_MB]->Fill(cent);
            }
          }
          else
          {
            if (hCentralityZOuterTrig[TrigIdx::TRIG14_MB])
            {
              hCentralityZOuterTrig[TrigIdx::TRIG14_MB]->Fill(cent);
            }
          }
        }
      }
    }
  }

  // skip event if zvtx is too large
  if (!m_pass_Zvtx)
  {
    if (Verbosity() > 0)
    {
      std::cout << "EventQA::process_centrality - [Event " << m_data.event << "] REJECTED: |zvtx| = " << std::abs(m_data.zvtx) << " cm >= " << m_cuts.m_zvtx_max << " cm" << std::endl;
    }
    ++m_ctr["process_eventCheck_zvtx_large"];
    return (m_doAbort) ? Fun4AllReturnCodes::ABORTEVENT : Fun4AllReturnCodes::EVENT_OK;
  }

  // skip event if not minimum bias
  if (!m_pass_MB)
  {
    return (m_doAbort) ? Fun4AllReturnCodes::ABORTEVENT : Fun4AllReturnCodes::EVENT_OK;
  }

  if (m_do_hist)
  {
    hEvent->Fill(static_cast<std::uint8_t>(EventType::MB));
  }

  // skip event if centrality is too peripheral
  if (cent >= m_cuts.m_cent_max)
  {
    if (Verbosity() > 0)
    {
      std::cout << "EventQA::process_centrality - [Event " << m_data.event << "] REJECTED: Centrality = " << cent << "% >= " << m_cuts.m_cent_max << "%" << std::endl;
    }
    ++m_ctr["process_eventCheck_centrality_large"];
    return (m_doAbort) ? Fun4AllReturnCodes::ABORTEVENT : Fun4AllReturnCodes::EVENT_OK;
  }

  if (m_do_hist)
  {
    hEvent->Fill(static_cast<std::uint8_t>(EventType::CENT));
  }

  if (Verbosity() > 0)
  {
    std::cout << "EventQA::process_centrality - [Event " << m_data.event << "] PASSED centrality cut" << std::endl;
  }

  return Fun4AllReturnCodes::EVENT_OK;
}

int EventQA::process_event(PHCompositeNode *topNode)
{
  int ret = process_event_check(topNode);
  if (ret && m_doAbort)
  {
    return ret;
  }

  ret = process_centrality(topNode);
  if (ret && m_doAbort)
  {
    return ret;
  }

  return Fun4AllReturnCodes::EVENT_OK;
}

int EventQA::ResetEvent([[maybe_unused]] PHCompositeNode *topNode)
{
  ++m_ctr["event_reset"];

  // Event
  m_data.run = 0;
  m_data.event = 0;
  m_data.zvtx = 9999;
  m_data.centrality = 9999;

  m_pass_MB = false;
  m_pass_Zvtx = false;
  m_didTrig12Fire = false;
  m_didTrig14Fire = false;

  return Fun4AllReturnCodes::EVENT_OK;
}

int EventQA::End([[maybe_unused]] PHCompositeNode *topNode)
{
  std::cout << "EventQA::End" << std::endl;

  std::cout << std::format("{:#<20}\n", "");
  std::cout << "stats" << std::endl;

  std::cout << std::format("{:#<20}\n", "");
  std::cout << "Abort Events Types" << std::endl;
  std::cout << std::format("process event, Reset Event Calls : {}", m_ctr["event_reset"]) << std::endl;
  std::cout << std::format("process event, MBD Trigger Fail: {}", m_ctr["process_eventCheck_mbd_trigger_fail"]) << std::endl;
  std::cout << std::format("process event, isAuAuMinBias Fail: {}", m_ctr["process_eventCheck_isAuAuMinBias_fail"]) << std::endl;
  std::cout << std::format("process event, |z| >= {} cm: {}", m_cuts.m_zvtx_max, m_ctr["process_eventCheck_zvtx_large"]) << std::endl;
  std::cout << std::format("process event, Centrality >= {}%: {}", m_cuts.m_cent_max, m_ctr["process_eventCheck_centrality_large"]) << std::endl;

  if (m_do_hist && hEvent)
  {
    std::cout << std::format("{:#<20}\n", "");
    std::cout << "Events" << std::endl;
    for (unsigned int i = 0; i < m_eventType.size(); ++i)
    {
      std::cout << m_eventType[i] << ": " << hEvent->GetBinContent(i + 1) << std::endl;
    }
    std::cout << std::format("{:#<20}\n", "");
  }

  return Fun4AllReturnCodes::EVENT_OK;
}
