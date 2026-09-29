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
    hEventTrigger = new TH1F("hEventTrigger", "Event Selection; Type; Events", static_cast<unsigned int>(m_eventTriggerType.size()), 0, static_cast<double>(m_eventTriggerType.size()));
    for (unsigned int i = 0; i < m_eventTriggerType.size(); ++i)
    {
      hEventTrigger->GetXaxis()->SetBinLabel(i + 1, m_eventTriggerType[i].c_str());
    }
    se->registerHisto(hEventTrigger);

    hVtxZ = new TH1F("hVtxZ", "Z Vertex; z [cm]; Events", m_hist_config.m_bins_zvtx, m_hist_config.m_zvtx_low, m_hist_config.m_zvtx_high);
    se->registerHisto(hVtxZ);

    hVtxZ_MB = new TH1F("hVtxZ_MB", "Z Vertex; z [cm]; Events", m_hist_config.m_bins_zvtx, m_hist_config.m_zvtx_low, m_hist_config.m_zvtx_high);
    se->registerHisto(hVtxZ_MB);

    hZVertex = new TH1F("hZVertex", "Min Bias; Z [cm]; Events", m_hist_config.m_bins_zvtx, m_hist_config.m_zvtx_low, m_hist_config.m_zvtx_high);
    se->registerHisto(hZVertex);

    // Combined Trigger 12 or 14 Vertex
    hZVertex_Trig12_or_Trig14 = new TH1F("hZVertex_Trig12_or_Trig14", "Trig 12 | Trig 14; Z [cm]; Events", m_hist_config.m_bins_zvtx, m_hist_config.m_zvtx_low, m_hist_config.m_zvtx_high);
    se->registerHisto(hZVertex_Trig12_or_Trig14);

    hZVertex_Trig12_or_Trig14_MB = new TH1F("hZVertex_Trig12_or_Trig14_MB", "Trig 12 | Trig 14 and MB; Z [cm]; Events", m_hist_config.m_bins_zvtx, m_hist_config.m_zvtx_low, m_hist_config.m_zvtx_high);
    se->registerHisto(hZVertex_Trig12_or_Trig14_MB);

    // Luminosity Histogram
    std::vector<std::string> lumiType{"|z| < 10 cm & MBD Trig", "|z| < 10 cm", "|z| < 10 cm & Trig 12", "|z| < 10 cm & Trig 14"};
    hLuminosity = new TH1F("hLuminosity", "; Type; Luminosity [nb^{-1}]", lumiType.size(), 0, lumiType.size());
    for (unsigned int i = 0; i < lumiType.size(); ++i)
    {
      hLuminosity->GetXaxis()->SetBinLabel(i + 1, lumiType[i].c_str());
    }
    se->registerHisto(hLuminosity);

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
      std::string title_h1 = std::format("{}; Z [cm]; Events", trig);
      std::string title_h2 = std::format("{}; Z [cm]; Centrality [%]", trig);

      std::string name_h1 = std::format("hZVertex_Trig{}", triggerIdx);
      std::string name_h2 = std::format("h2ZVertexCentrality_Trig{}", triggerIdx);

      hZVertexTrig[idx_relaxed] = new TH1F(name_h1.c_str(), title_h1.c_str(), m_hist_config.m_bins_zvtx, m_hist_config.m_zvtx_low, m_hist_config.m_zvtx_high);
      h2ZVertexCentralityTrig[idx_relaxed] = new TH2F(name_h2.c_str(), title_h2.c_str(), m_hist_config.m_bins_zvtx, m_hist_config.m_zvtx_low, m_hist_config.m_zvtx_high, m_hist_config.m_bins_cent, m_hist_config.m_cent_low, m_hist_config.m_cent_high);

      se->registerHisto(hZVertexTrig[idx_relaxed]);
      se->registerHisto(h2ZVertexCentralityTrig[idx_relaxed]);

      // Tight (trigger + offline MB)
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
    if (hEventTrigger)
    {
      if (m_didTrig12Fire)
      {
        hEventTrigger->Fill(static_cast<std::uint8_t>(EventTriggerType::TRIG12));
      }
      if (m_didTrig14Fire)
      {
        hEventTrigger->Fill(static_cast<std::uint8_t>(EventTriggerType::TRIG14));
      }
      if (mbd_trigger_fire)
      {
        hEventTrigger->Fill(static_cast<std::uint8_t>(EventTriggerType::TRIG12_OR_TRIG14));
      }
      if (pass_zvtx10)
      {
        hEventTrigger->Fill(static_cast<std::uint8_t>(EventTriggerType::ZVTX10));
        if (m_didTrig12Fire)
        {
          hEventTrigger->Fill(static_cast<std::uint8_t>(EventTriggerType::ZVTX10_TRIG12));
        }
        if (m_didTrig14Fire)
        {
          hEventTrigger->Fill(static_cast<std::uint8_t>(EventTriggerType::ZVTX10_TRIG14));
        }
        if (mbd_trigger_fire)
        {
          hEventTrigger->Fill(static_cast<std::uint8_t>(EventTriggerType::ZVTX10_TRIG12_OR_TRIG14));
        }
      }
    }

    if (!vertexmap->empty())
    {
      if (mbd_trigger_fire && hZVertex_Trig12_or_Trig14)
      {
        hZVertex_Trig12_or_Trig14->Fill(zvtx);
      }
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

  // Update luminosity event counters
  if (pass_zvtx10)
  {
    ++m_n_zvtx10;
    if (mbd_trigger_fire)
    {
      ++m_n_zvtx10_trig_or;
    }
    if (m_didTrig12Fire)
    {
      ++m_n_zvtx10_trig12;
    }
    if (m_didTrig14Fire)
    {
      ++m_n_zvtx10_trig14;
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
  if (m_check_mb)
  {
    MinimumBiasInfo *m_mb_info = findNode::getClass<MinimumBiasInfo>(topNode, "MinimumBiasInfo");
    PdbParameterMap *pdb = findNode::getClass<PdbParameterMap>(topNode, "MinBiasParams");

    if (!m_mb_info || !pdb)
    {
      if (m_strict_node_check)
      {
        std::cout << "Aborting Run: MinimumBiasInfo or MinBiasParams null" << std::endl;
        return Fun4AllReturnCodes::ABORTRUN;
      }
      static bool warned_mb = false;
      if (!warned_mb)
      {
        std::cout << "EventQA: MinimumBiasInfo or MinBiasParams not found on node tree. Skipping offline MB checks." << std::endl;
        warned_mb = true;
      }
      m_check_mb = false;
    }
    else
    {
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
          if (mbd_trigger_fire && hZVertex_Trig12_or_Trig14_MB)
          {
            hZVertex_Trig12_or_Trig14_MB->Fill(zvtx);
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

  if (Verbosity() > 0)
  {
    std::cout << "EventQA::process_event_check - [Event " << m_data.event << "] PASSED event selection" << std::endl;
  }

  return Fun4AllReturnCodes::EVENT_OK;
}

int EventQA::process_centrality(PHCompositeNode *topNode)
{
  if (!m_check_centrality)
  {
    return Fun4AllReturnCodes::EVENT_OK;
  }

  CentralityInfo *centInfo = findNode::getClass<CentralityInfo>(topNode, "CentralityInfo");
  if (!centInfo)
  {
    if (m_strict_node_check)
    {
      std::cout << "Aborting Run: CentralityInfo null" << std::endl;
      return Fun4AllReturnCodes::ABORTRUN;
    }
    static bool warned_cent = false;
    if (!warned_cent)
    {
      std::cout << "EventQA: CentralityInfo not found on node tree. Skipping centrality checks." << std::endl;
      warned_cent = true;
    }
    m_check_centrality = false;
    return Fun4AllReturnCodes::EVENT_OK;
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
      if (h2ZVertexCentrality)
      {
        h2ZVertexCentrality->Fill(m_data.zvtx, cent);
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

      // Tight: trigger + offline MB
      if (m_pass_MB)
      {
        if (h2ZVertexCentralityTrig[TrigIdx::TRIG12_MB])
        {
          h2ZVertexCentralityTrig[TrigIdx::TRIG12_MB]->Fill(m_data.zvtx, cent);
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

      // Tight: trigger + offline MB
      if (m_pass_MB)
      {
        if (h2ZVertexCentralityTrig[TrigIdx::TRIG14_MB])
        {
          h2ZVertexCentralityTrig[TrigIdx::TRIG14_MB]->Fill(m_data.zvtx, cent);
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
  if (m_check_mb && !m_pass_MB)
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
  ++m_total_events;

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

  int prescale_12 = (m_triggerAnalyzer) ? m_triggerAnalyzer->getTriggerPrescale(m_trig_12) : -1;
  int prescale_14 = (m_triggerAnalyzer) ? m_triggerAnalyzer->getTriggerPrescale(m_trig_14) : -1;

  std::cout << std::format("Trigger: {}, Prescale: {}\n", m_trig_12, prescale_12);
  std::cout << std::format("Trigger: {}, Prescale: {}\n", m_trig_14, prescale_14);

  if (m_do_hist && hLuminosity)
  {
    double lumi_trig = 0.0;
    double lumi_vtx = 0.0;
    double lumi_trig12 = 0.0;
    double lumi_trig14 = 0.0;

    if (prescale_12 > 0 || prescale_14 > 0)
    {
      lumi_trig = static_cast<double>(m_n_zvtx10_trig_or) / m_sigma_mbd * 1e-9;
    }

    if (prescale_12 > 0)
    {
      lumi_trig12 = static_cast<double>(m_n_zvtx10_trig12) / m_sigma_mbd * 1e-9;
    }

    if (prescale_14 > 0)
    {
      lumi_trig14 = static_cast<double>(m_n_zvtx10_trig14) / m_sigma_mbd * 1e-9;
    }

    // VTX-only calculation (ignores trigger prescale statuses)
    lumi_vtx = static_cast<double>(m_n_zvtx10) / m_sigma_mbd * 1e-9;

    hLuminosity->SetBinContent(1, lumi_trig);
    hLuminosity->SetBinContent(2, lumi_vtx);
    hLuminosity->SetBinContent(3, lumi_trig12);
    hLuminosity->SetBinContent(4, lumi_trig14);
  }

  std::cout << std::format("{:#<20}\n", "");
  std::cout << "stats" << std::endl;

  std::cout << std::format("{:#<20}\n", "");
  std::cout << "Abort Events Types" << std::endl;
  std::cout << std::format("process event, Total Event Calls: {}", m_total_events) << std::endl;
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
