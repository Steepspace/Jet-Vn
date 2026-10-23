#include "TruthJetValidation.h"

// -- c++
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numbers>

// -- event
#include <ffaobjects/EventHeader.h>

// -- Fun4All
#include <fun4all/Fun4AllInputManager.h>
#include <fun4all/Fun4AllReturnCodes.h>
#include <fun4all/Fun4AllServer.h>

// -- Nodes
#include <phool/PHCompositeNode.h>
#include <phool/getClass.h>

// -- jet
#include <jetbase/Jet.h>
#include <jetbase/JetContainer.h>

// -- generator
#include <phhepmc/PHGenIntegral.h>

#include <treefiller/TreeFiller.h>

// -- ROOT
#include <TH1.h>
#include <TTree.h>

//____________________________________________________________________________..
TruthJetValidation::TruthJetValidation(const std::string &name)
  : SubsysReco(name)
{
}

void TruthJetValidation::set_jet_radii(const std::vector<float>& radii)
{
  m_do_r02 = false;
  m_do_r03 = false;
  for (float r : radii)
  {
    if (std::abs(r - 0.2F) < 1e-3F)
    {
      m_do_r02 = true;
    }
    if (std::abs(r - 0.3F) < 1e-3F)
    {
      m_do_r03 = true;
    }
  }
}

//____________________________________________________________________________..
int TruthJetValidation::Init([[maybe_unused]] PHCompositeNode *topNode)
{
  TTree* tree = TreeFiller::getTree();
  if (!tree)
  {
    std::cout << "TruthJetValidation: Failed to get/create TTree" << std::endl;
    return Fun4AllReturnCodes::ABORTRUN;
  }

  // books the per-jet branches "<field>_truth_<radius>", e.g. pt_truth_r02
  auto book_jets = [&](JetData& jd, const std::string& radius)
  {
    std::string tag = "truth_" + radius;
    tree->Branch(("max_pt_" + tag).c_str(), &jd.max_pt);
    tree->Branch(("pt_" + tag).c_str(), &jd.pt);
    tree->Branch(("e_" + tag).c_str(), &jd.e);
    tree->Branch(("phi_" + tag).c_str(), &jd.phi);
    tree->Branch(("eta_" + tag).c_str(), &jd.eta);
  };

  if (m_do_r02)
  {
    book_jets(m_truth_r02, "r02");
  }
  if (m_do_r03)
  {
    book_jets(m_truth_r03, "r03");
  }
  if (m_do_max_pt_r04)
  {
    tree->Branch("max_pt_truth_r04", &m_max_pt_truth_r04);
  }

  // generator bookkeeping, saved with the other histograms (hist file)
  m_h_gen_integral = new TH1D("h_gen_integral", "Generator Bookkeeping", kNFiles, 0.5, static_cast<double>(kNFiles) + 0.5);
  m_h_gen_integral->GetXaxis()->SetBinLabel(kLumiFile, "IntegratedLumi_file [pb^{-1}]");
  m_h_gen_integral->GetXaxis()->SetBinLabel(kNProcessedFile, "NProcessed_file");
  m_h_gen_integral->GetXaxis()->SetBinLabel(kNGenAcceptedFile, "NGeneratorAccepted_file");
  m_h_gen_integral->GetXaxis()->SetBinLabel(kSumWeightFile, "SumOfWeight_file");
  m_h_gen_integral->GetXaxis()->SetBinLabel(kNEventsJob, "NEvents_job");
  m_h_gen_integral->GetXaxis()->SetBinLabel(kLumiJob, "IntegratedLumi_job [pb^{-1}]");
  m_h_gen_integral->GetXaxis()->SetBinLabel(kNFiles, "NFiles");
  Fun4AllServer::instance()->registerHisto(m_h_gen_integral);

  return Fun4AllReturnCodes::EVENT_OK;
}

//____________________________________________________________________________..
int TruthJetValidation::InitRun(PHCompositeNode *topNode)
{
  PHGenIntegral *gen_integral = findNode::getClass<PHGenIntegral>(topNode, "PHGenIntegral");
  if (!gen_integral)
  {
    std::cout << "TruthJetValidation: PHGenIntegral null, h_gen_integral is not filled" << std::endl;
    return Fun4AllReturnCodes::EVENT_OK;
  }

  m_lumi_file = gen_integral->get_Integrated_Lumi();
  m_nprocessed_file = static_cast<double>(gen_integral->get_N_Processed_Event());

  m_h_gen_integral->AddBinContent(kLumiFile, m_lumi_file);
  m_h_gen_integral->AddBinContent(kNProcessedFile, m_nprocessed_file);
  m_h_gen_integral->AddBinContent(kNGenAcceptedFile, static_cast<double>(gen_integral->get_N_Generator_Accepted_Event()));
  m_h_gen_integral->AddBinContent(kSumWeightFile, gen_integral->get_Sum_Of_Weight());
  m_h_gen_integral->AddBinContent(kNFiles, 1);

  std::cout << "TruthJetValidation: " << gen_integral->get_Description()
            << " | lumi: " << m_lumi_file << " pb^-1"
            << " | processed: " << gen_integral->get_N_Processed_Event()
            << " | generator accepted: " << gen_integral->get_N_Generator_Accepted_Event() << std::endl;

  return Fun4AllReturnCodes::EVENT_OK;
}

//____________________________________________________________________________..
void TruthJetValidation::fill_jets(JetContainer *jets, JetData &jd) const
{
  for (auto *jet : *jets)
  {
    if (!jet)
    {
      continue;
    }

    double pt = jet->get_pt();
    double eta = jet->get_eta();
    double phi = jet->get_phi();

    // map [-pi,pi] -> [0,2pi]
    if (phi < 0)
    {
      phi += 2.0 * std::numbers::pi;
    }

    // loose acceptance only: the fiducial cut is applied in the analysis
    if (pt < m_jet_pt_min_cut || std::abs(eta) > m_jet_eta_max_cut)
    {
      continue;
    }

    jd.pt.push_back(pt);
    jd.e.push_back(jet->get_e());
    jd.phi.push_back(phi);
    jd.eta.push_back(eta);
    jd.max_pt = std::max(jd.max_pt, pt);
  }
}

//____________________________________________________________________________..
int TruthJetValidation::process_event(PHCompositeNode *topNode)
{
  EventHeader *eventInfo = findNode::getClass<EventHeader>(topNode, "EventHeader");
  int event_id = eventInfo ? eventInfo->get_EvtSequence() : -1;

  // the overlay MC inputs are read in file order: they must stay on the data event they were embedded into
  if (!m_data_topnode.empty())
  {
    Fun4AllServer *se = Fun4AllServer::instance();
    PHCompositeNode *dataTopNode = se->topNode(m_data_topnode);
    EventHeader *dataEventInfo = findNode::getClass<EventHeader>(dataTopNode, "EventHeader");
    int data_event_id = dataEventInfo ? dataEventInfo->get_EvtSequence() : -1;

    // MC ahead of data: the overlay production dropped this data event (e.g. no vertex, nothing embedded).
    // Skip it, and re-read the current MC event together with the next data event.
    if (eventInfo && dataEventInfo && event_id > data_event_id && !m_mc_input_managers.empty())
    {
      for (const auto &name : m_mc_input_managers)
      {
        Fun4AllInputManager *in = se->getInputManager(name);
        if (!in || in->PushBackEvents(1) != 0)
        {
          std::cout << "Aborting Run: could not push back MC input manager " << name << std::endl;
          return Fun4AllReturnCodes::ABORTRUN;
        }
      }
      ++m_nskipped_data;
      if (Verbosity() > 0)
      {
        std::cout << "TruthJetValidation: data event " << data_event_id << " has no MC event (next MC event "
                  << event_id << "), skipped" << std::endl;
      }
      return Fun4AllReturnCodes::ABORTEVENT;
    }

    if (!eventInfo || !dataEventInfo || data_event_id != event_id)
    {
      std::cout << "Aborting Run: MC event " << event_id << " does not match data event "
                << data_event_id << std::endl;
      return Fun4AllReturnCodes::ABORTRUN;
    }
  }

  ++m_nevents_job;

  if (m_do_r02)
  {
    JetContainer *jets_r02 = findNode::getClass<JetContainer>(topNode, m_truthJetName_r02);
    if (!jets_r02)
    {
      std::cout << "Aborting Run: " << m_truthJetName_r02 << " null" << std::endl;
      return Fun4AllReturnCodes::ABORTRUN;
    }
    fill_jets(jets_r02, m_truth_r02);
  }

  if (m_do_r03)
  {
    JetContainer *jets_r03 = findNode::getClass<JetContainer>(topNode, m_truthJetName_r03);
    if (!jets_r03)
    {
      std::cout << "Aborting Run: " << m_truthJetName_r03 << " null" << std::endl;
      return Fun4AllReturnCodes::ABORTRUN;
    }
    fill_jets(jets_r03, m_truth_r03);
  }

  // no pT or eta cut: this is the variable the generator trigger cut on
  if (m_do_max_pt_r04)
  {
    JetContainer *jets_r04 = findNode::getClass<JetContainer>(topNode, m_truthJetName_r04);
    if (!jets_r04)
    {
      std::cout << "Aborting Run: " << m_truthJetName_r04 << " null" << std::endl;
      return Fun4AllReturnCodes::ABORTRUN;
    }
    for (auto *jet : *jets_r04)
    {
      if (jet)
      {
        m_max_pt_truth_r04 = std::max(m_max_pt_truth_r04, static_cast<double>(jet->get_pt()));
      }
    }
  }

  if (Verbosity() > 0)
  {
    std::cout << "TruthJetValidation::process_event - [Event " << event_id << "]"
              << " | R=0.2 jets: " << m_truth_r02.pt.size() << " (max pT = " << m_truth_r02.max_pt << ")"
              << " | R=0.3 jets: " << m_truth_r03.pt.size() << " (max pT = " << m_truth_r03.max_pt << ")"
              << " | max pT R=0.4: " << m_max_pt_truth_r04 << std::endl;
  }

  // TTree filling is managed by TreeFiller

  return Fun4AllReturnCodes::EVENT_OK;
}

//____________________________________________________________________________..
int TruthJetValidation::ResetEvent([[maybe_unused]] PHCompositeNode *topNode)
{
  m_truth_r02.clear();
  m_truth_r03.clear();
  m_max_pt_truth_r04 = 0;

  return Fun4AllReturnCodes::EVENT_OK;
}

//____________________________________________________________________________..
int TruthJetValidation::End([[maybe_unused]] PHCompositeNode *topNode)
{
  // luminosity of the events analyzed by this job: the file luminosity scaled to the analyzed fraction
  double lumi_job = (m_nprocessed_file > 0) ? m_lumi_file * m_nevents_job / m_nprocessed_file : 0;
  m_h_gen_integral->AddBinContent(kNEventsJob, m_nevents_job);
  m_h_gen_integral->AddBinContent(kLumiJob, lumi_job);

  std::cout << "TruthJetValidation::End - events analyzed: " << m_nevents_job
            << " | lumi: " << lumi_job << " pb^-1"
            << " | data events skipped (no MC): " << m_nskipped_data << std::endl;

  return Fun4AllReturnCodes::EVENT_OK;
}
