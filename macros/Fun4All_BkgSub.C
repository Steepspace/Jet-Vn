// c++ includes --
#include <algorithm>
#include <map>
#include <string>
#include <iostream>
#include <filesystem>
#include <vector>
#include <format>
#include <fstream>

// root includes --
#include <TSystem.h>
#include <TROOT.h>
#include <TF1.h>


#include <mbd/MbdReco.h>
#include <mbd/MbdEvent.h>
#include <epd/EpdReco.h>
#include <globalvertex/GlobalVertexReco.h>
#include <centrality/CentralityReco.h>
#include <calotrigger/MinimumBiasClassifier.h>

#include <ffamodules/CDBInterface.h>
#include <ffamodules/FlagHandler.h>

#include <fun4all/Fun4AllHistoManager.h>
#include <fun4all/Fun4AllDstInputManager.h>
#include <fun4all/Fun4AllDstOutputManager.h>
#include <fun4all/Fun4AllInputManager.h>
#include <fun4all/Fun4AllNoSyncDstInputManager.h>
#include <fun4all/Fun4AllServer.h>
#include <fun4all/Fun4AllBase.h>
#include <fun4all/Fun4AllUtils.h>

#include <phool/recoConsts.h>

#include <calotrigger/TriggerRunInfoReco.h>

#include <sepdvalidation/EventSkip.h>
#include <sepdvalidation/EventQA.h>
#include <sepdvalidation/GlobalQA.h>
#include <sepdvalidation/CaloQA.h>
#include <sepdvalidation/JetValidationv3.h>
#include <sepdvalidation/RandomConeValidation.h>
#include <sepdvalidation/TruthJetValidation.h>

#include <treefiller/TreeFiller.h>

#include "Calo_Calib.C"
#include "HIJetReco.C"

R__LOAD_LIBRARY(libg4detectors_io.so)
R__LOAD_LIBRARY(libsEPDValidation.so)
R__LOAD_LIBRARY(libTreeFiller.so)

void Fun4All_BkgSub(const std::string &flist_dst_calofit = "DST_CALOFITTING_run3auau_pro001_pcdb001_v001-00068144-00000.root",
                    const std::string &flist_dst_zdc = "/direct/sphenix+tg+tg01/jets/anarde/run3auau/ZDC/68144/DST_ZDC_CALIB_run3auau_pro001_pcdb001_v001-00068144-00000.root",
                    const std::string &flist_dst_sepd = "/direct/sphenix+tg+tg01/jets/anarde/run3auau/sEPD/68144/DST_SEPD_CALIB_run3auau_pro001_pcdb001_v001-00068144-00000.root",
                    const std::string &input_QVecCalib = "/sphenix/user/anarde/sEPD-Calib/run3auau/10-04-26-v2/CDB/68144/SEPD_EventPlaneCalib-pro001_pcdb001_v001-68144v2.root",
                    const std::string &output = "test.root",
                    const std::string &output_tree = "tree.root",
                    int do_flow = 3,
                    int nEvents = 100,
                    int nSkip = 0,
                    int event_id = 0,
                    const std::string &dbtag = "newcdbtag",
                    const std::string &eta_calib_direct_path = "/direct/sphenix+u/anarde/Documents/sPHENIX/Jet-Vn/files/calib/rho_eta_calib_cdb_signed.root",
                    const std::string &event_list = "",
                    bool do_rcone = false,
                    bool do_mult = false,
                    const std::string &output_dst = "",
                    bool do_neg_energy_threshold = true,
                    float neg_energy_threshold = -2.0F,
                    const std::string &zs_calib_cemc = "",
                    const std::string &zs_calib_hcalin = "",
                    const std::string &zs_calib_hcalout = "",
                    bool do_emfrac_calib = false,
                    const std::string &emfrac_calib_path = "",
                    const std::string &overlay_dst_calo = "",
                    const std::string &overlay_dst_truth_jet = "",
                    float jet_pt_min = 10.0F,
                    float truth_jet_pt_min = 5.0F)
{
  // data+MC overlay: flist_dst_calofit is the data-only DST_CALOFITTING of the overlay sample
  // (event aligned with the overlay DSTs), flist_dst_zdc/sepd the data ZDC/sEPD DSTs of the same
  // segment. The calo towers are swapped for the embedded MC+data towers of DST_CALO, everything
  // else is reconstructed exactly as in data.
  bool is_overlay = !overlay_dst_calo.empty() && overlay_dst_calo != "none";
  bool is_single_dst = flist_dst_zdc.empty() || flist_dst_zdc == "none";

  // Extract runnumber and segment from first file within list
  int runnumber = 0;
  int segment = 0;
  bool isFileList = true;
  // single file
  if (flist_dst_calofit.ends_with(".root"))
  {
    std::pair<int, int> runseg = Fun4AllUtils::GetRunSegment(flist_dst_calofit);
    runnumber = runseg.first;
    segment = runseg.second;
    isFileList = false;
  }
  // list of files
  else
  {
    std::ifstream infile_stream(flist_dst_calofit);
    if (!infile_stream) {
      std::cout << "Error: Could not open file list " << flist_dst_calofit << std::endl;
      return;
    }
    std::string filepath;
    getline(infile_stream, filepath);
    std::pair<int, int> runseg = Fun4AllUtils::GetRunSegment(filepath);
    runnumber = runseg.first;
    segment = runseg.second;
    infile_stream.close();
  }

  // ZDC and sEPD inputs must come from the same run as calofit
  // (the default paths are for run 68144)
  if (!is_single_dst)
  {
    const std::vector<std::pair<std::string, std::string>> run_inputs = {
        {"zdc", flist_dst_zdc},
        {"sepd", flist_dst_sepd}};

    for (const auto& [name, path] : run_inputs)
    {
      // same list-vs-single-file handling as the calofit input
      std::string filepath = path;
      if (isFileList)
      {
        std::ifstream list_stream(path);
        std::getline(list_stream, filepath);
      }

      int input_run = Fun4AllUtils::GetRunSegment(filepath).first;
      if (input_run != runnumber)
      {
        std::cout << "Error: " << name << " input run " << input_run
                  << " does not match calofit run " << runnumber << std::endl;
        return;
      }
    }
  }

  if (is_overlay)
  {
    if (isFileList)
    {
      std::cout << "Error: overlay mode takes single files, not lists: " << flist_dst_calofit << std::endl;
      return;
    }
    if (is_single_dst)
    {
      std::cout << "Error: overlay mode needs the data ZDC and sEPD DSTs of the segment" << std::endl;
      return;
    }
    // the prepared DST_CALOFITTING holds the events of one data segment
    for (const auto& path : {flist_dst_zdc, flist_dst_sepd})
    {
      int input_segment = Fun4AllUtils::GetRunSegment(path).second;
      if (input_segment != segment)
      {
        std::cout << "Error: " << path << " segment " << input_segment
                  << " does not match calofit segment " << segment << std::endl;
        return;
      }
    }
    if (overlay_dst_truth_jet.empty() || overlay_dst_truth_jet == "none")
    {
      std::cout << "Error: overlay mode needs the DST_TRUTH_JET input" << std::endl;
      return;
    }
    // the flow and no flow iterative chains are only different with flow modulation
    if (do_flow == 0)
    {
      std::cout << "Error: overlay mode needs do_flow != 0" << std::endl;
      return;
    }

    const std::vector<std::pair<std::string, std::string>> overlay_inputs = {
        {"calo", overlay_dst_calo},
        {"truth jet", overlay_dst_truth_jet}};

    for (const auto& [name, path] : overlay_inputs)
    {
      int input_run = Fun4AllUtils::GetRunSegment(path).first;
      if (input_run != runnumber)
      {
        std::cout << "Error: overlay " << name << " input run " << input_run
                  << " does not match calofit run " << runnumber << std::endl;
        return;
      }
    }

    // only the iterative jets (with and without flow modulation) are needed for the overlay
    do_mult = false;
  }

  // ZS cross calib T0 files must also match the run (the defaults are for run 68144)
  // "none" or empty: no direct URL, CaloTowerCalib falls back to the CDB
  const std::vector<std::pair<std::string, std::string>> zs_calibs = {
      {"CEMC", zs_calib_cemc},
      {"HCALIN", zs_calib_hcalin},
      {"HCALOUT", zs_calib_hcalout}};

  std::map<std::string, std::string> zs_urls;
  for (const auto& [name, path] : zs_calibs)
  {
    if (path.empty() || path == "none")
    {
      std::cout << "No ZS cross calib T0 for " << name << ", using CDB" << std::endl;
      continue;
    }

    // file names end in _<run>.root
    std::string stem = std::filesystem::path(path).stem().string();
    std::string run_str = stem.substr(stem.find_last_of('_') + 1);
    int calib_run = (!run_str.empty() && std::all_of(run_str.begin(), run_str.end(), ::isdigit)) ? std::stoi(run_str) : -1;
    if (calib_run != runnumber)
    {
      std::cout << "Error: " << name << " ZS cross calib T0 run " << calib_run
                << " does not match calofit run " << runnumber << ": " << path << std::endl;
      return;
    }
    zs_urls[name] = path;
  }

  std::cout << "########################" << std::endl;
  std::cout << "Run Parameters" << std::endl;
  if (is_overlay)
  {
    std::cout << "overlay mode" << std::endl;
    std::cout << "input calofit (data): " << flist_dst_calofit << std::endl;
    std::cout << "input zdc: " << flist_dst_zdc << std::endl;
    std::cout << "input sepd: " << flist_dst_sepd << std::endl;
    std::cout << "input calo (data+MC): " << overlay_dst_calo << std::endl;
    std::cout << "input truth jet: " << overlay_dst_truth_jet << std::endl;
  }
  else if (is_single_dst)
  {
    std::cout << "input single DST: " << flist_dst_calofit << std::endl;
  }
  else
  {
    std::cout << "input calofit: " << flist_dst_calofit << std::endl;
    std::cout << "input zdc: " << flist_dst_zdc << std::endl;
    std::cout << "input sepd: " << flist_dst_sepd << std::endl;
  }
  std::cout << "Run: " << runnumber << std::endl;
  std::cout << "QVec Calib: " << input_QVecCalib << std::endl;
  std::cout << "output: " << output << std::endl;
  std::cout << "output tree: " << output_tree << std::endl;
  std::cout << "output dst: " << output_dst << std::endl;
  std::cout << "do_flow: " << do_flow << std::endl;
  std::cout << "nEvents: " << nEvents << std::endl;
  std::cout << "nSkip: " << nSkip << std::endl;
  std::cout << "event_id: " << event_id << std::endl;
  std::cout << "event_list: " << event_list << std::endl;
  std::cout << "dbtag: " << dbtag << std::endl;
  std::cout << "eta_calib_direct_path: " << eta_calib_direct_path << std::endl;
  std::cout << "do_rcone: " << do_rcone << std::endl;
  std::cout << "do_mult: " << do_mult << std::endl;
  std::cout << "do_neg_energy_threshold: " << do_neg_energy_threshold << std::endl;
  std::cout << "neg_energy_threshold: " << neg_energy_threshold << std::endl;
  std::cout << "zs_calib_cemc: " << zs_calib_cemc << std::endl;
  std::cout << "zs_calib_hcalin: " << zs_calib_hcalin << std::endl;
  std::cout << "zs_calib_hcalout: " << zs_calib_hcalout << std::endl;
  std::cout << "do_emfrac_calib: " << do_emfrac_calib << std::endl;
  std::cout << "emfrac_calib_path: " << (emfrac_calib_path.empty() ? "CDB" : emfrac_calib_path) << std::endl;
  std::cout << "jet_pt_min: " << jet_pt_min << std::endl;
  if (is_overlay)
  {
    std::cout << "truth_jet_pt_min: " << truth_jet_pt_min << std::endl;
  }
  std::cout << "########################" << std::endl;

  Fun4AllServer *se = Fun4AllServer::instance();
  se->Verbosity(Fun4AllBase::VERBOSITY_SOME);
  se->VerbosityDownscale(5000);

  recoConsts *rc = recoConsts::instance();

  // conditions DB flags and timestamp
  rc->set_StringFlag("CDB_GLOBALTAG", dbtag);
  rc->set_uint64Flag("TIMESTAMP", runnumber);
  CDBInterface::instance()->Verbosity(Fun4AllBase::VERBOSITY_SOME);

  FlagHandler* flag = new FlagHandler();
  se->registerSubsystem(flag);

  // Event Skip
  EventSkip* evtSkip = new EventSkip();
  evtSkip->set_skip(nSkip);
  if (!event_list.empty())
  {
    evtSkip->set_event_ids(event_list);
  }
  else if (event_id != 0)
  {
    evtSkip->set_event_id(event_id);
  }
  evtSkip->Verbosity(Fun4AllBase::VERBOSITY_QUIET);
  se->registerSubsystem(evtSkip);

  // Truth Jets (overlay only), on the MC top node: checks that each MC event matches the data
  // event and counts the analyzed events for the luminosity, so it runs before any event selection
  if (is_overlay)
  {
    TruthJetValidation* truth_jet_validation = new TruthJetValidation();
    truth_jet_validation->set_jet_radii({0.2F, 0.3F});
    truth_jet_validation->set_jet_pt_min(truth_jet_pt_min);
    truth_jet_validation->set_data_topnode("TOP");
    // data events dropped by the overlay production are skipped by pushing back the MC inputs
    truth_jet_validation->set_mc_input_managers({"calo", "truthjet"});
    truth_jet_validation->Verbosity(Fun4AllBase::VERBOSITY_QUIET);
    se->registerSubsystem(truth_jet_validation, "TOPMC");
  }

  // Calibrate Towers
  CaloCalib::do_neg_energy_threshold = do_neg_energy_threshold;
  CaloCalib::neg_energy_threshold = neg_energy_threshold;
  CaloCalib::cemc_ZScrosscalibT0_url = zs_urls["CEMC"];
  CaloCalib::hcalin_ZScrosscalibT0_url = zs_urls["HCALIN"];
  CaloCalib::hcalout_ZScrosscalibT0_url = zs_urls["HCALOUT"];
  Process_Calo_Calib();

  // MBD Reconstruction
  MbdReco* mbdreco = new MbdReco();
  se->registerSubsystem(mbdreco);

  // sEPD Reconstruction--Calib Info
  if (do_flow == 3)
  {
    EpdReco* epdreco = new EpdReco();
    se->registerSubsystem(epdreco);
  }

  // Official vertex storage
  GlobalVertexReco* gvertex = new GlobalVertexReco();
  gvertex->Verbosity(Fun4AllBase::VERBOSITY_QUIET);
  // MBD only: the default (UNDEFINED) also adds a z = 0 placeholder vertex every event
  gvertex->setVertexType(GlobalVertex::VTXTYPE::MBD);
  se->registerSubsystem(gvertex);

  // Trigger Info Reco
  TriggerRunInfoReco* trig = new TriggerRunInfoReco();
  trig->Verbosity(1);
  se->registerSubsystem(trig);

  // Minimum Bias Classifier
  MinimumBiasClassifier* mb = new MinimumBiasClassifier();
  se->registerSubsystem(mb);

  // Centrality Reco
  CentralityReco* cent = new CentralityReco();
  se->registerSubsystem(cent);

  bool do_detailed = !event_list.empty() || event_id != 0;

  // Event QA
  EventQA* event_qa = new EventQA();
  if (do_detailed)
  {
    event_qa->set_do_hist(false);
  }
  event_qa->set_cent_max(80);
  event_qa->Verbosity(Fun4AllBase::VERBOSITY_QUIET);
  se->registerSubsystem(event_qa);

  // Jet Reco
  // Enable::HIJETS_VERBOSITY = 10;
  HIJETS::eventplane_custom_calib = input_QVecCalib;
  HIJETS::do_flow = do_flow;
  if (eta_calib_direct_path != "none" && !eta_calib_direct_path.empty())
  {
    HIJETS::eta_calib_direct_path = eta_calib_direct_path;
  }
  Enable::HIJETS_TOWER_MULTSUB = do_mult;
  Enable::HIJETS_TOWER_NOBKG = !is_overlay;
  // with do_flow == 0 the default iterative/mult chains are already no-flow
  Enable::HIJETS_TOWER_NOFLOW = (do_flow != 0);
  Enable::HIJETS_TOWER_MULTSUB_NOFLOW = do_mult && (do_flow != 0);
  // side-by-side EMfrac JES (legacy calib nodes are unchanged)
  Enable::HIJETS_TOWER_CALIB_EMFRAC = do_emfrac_calib;
  HIJETS::emfrac_calib_direct_path = emfrac_calib_path;
  HIJetReco();

  // Calo QA
  CaloQA* calo_qa = new CaloQA();
  calo_qa->set_do_hist(false);
  calo_qa->set_do_detailed(do_detailed);
  calo_qa->set_do_iter(do_detailed);
  calo_qa->set_do_mult(do_detailed && do_mult);
  calo_qa->Verbosity(Fun4AllBase::VERBOSITY_QUIET);
  se->registerSubsystem(calo_qa);

  // Global QA
  GlobalQA* global_qa = new GlobalQA();
  global_qa->set_do_ep(do_flow);
  global_qa->set_do_hist(false);
  global_qa->Verbosity(Fun4AllBase::VERBOSITY_QUIET);
  se->registerSubsystem(global_qa);

  // Jet Validation
  JetValidationv3* jet_validation = new JetValidationv3();
  jet_validation->set_do_unsub(Enable::HIJETS_TOWER_NOBKG);
  jet_validation->set_do_mult(do_mult);
  jet_validation->set_do_iter_noflow(Enable::HIJETS_TOWER_NOFLOW);
  jet_validation->set_do_mult_noflow(Enable::HIJETS_TOWER_MULTSUB_NOFLOW);
  jet_validation->set_do_detailed(do_detailed);
  jet_validation->set_do_emfrac(do_emfrac_calib);
  jet_validation->set_jet_pt_min(jet_pt_min);
  jet_validation->Verbosity(Fun4AllBase::VERBOSITY_QUIET);
  se->registerSubsystem(jet_validation);

  // Random Cone Validation
  if (do_rcone)
  {
    RandomConeValidation* random_cone_validation = new RandomConeValidation();
    random_cone_validation->set_do_detailed(do_detailed);
    random_cone_validation->Verbosity(Fun4AllBase::VERBOSITY_QUIET);
    se->registerSubsystem(random_cone_validation);
  }

  // Tree Filler
  TreeFiller* tree_filler = TreeFiller::instance();
  tree_filler->setOutfileName(output_tree);
  tree_filler->Verbosity(Fun4AllBase::VERBOSITY_QUIET);
  se->registerSubsystem(tree_filler);

  // Output DST for detailed events (event_id or event_list)
  if (do_detailed && output_dst != "none")
  {
    std::string dst_name = output_dst;
    if (dst_name.empty())
    {
      dst_name = std::format("dst-{:08d}-{:05d}.root", runnumber, segment);
    }

    Fun4AllDstOutputManager *dst_out = new Fun4AllDstOutputManager("DSTOUT", dst_name);
    dst_out->SplitLevel(99);
    dst_out->AddNode("Sync");
    dst_out->AddNode("EventHeader");
    dst_out->AddNode("14001");
    dst_out->AddNode("MbdRawContainer");
    dst_out->AddNode("TOWERS_CEMC");
    dst_out->AddNode("TOWERS_HCALIN");
    dst_out->AddNode("TOWERS_HCALOUT");
    dst_out->AddNode("TOWERS_SEPD");
    dst_out->AddNode("Zdcinfo");
    se->registerOutputManager(dst_out);
  }

  if (is_overlay)
  {
    // Data side, synced as in data (run 68144): the prepared DST_CALOFITTING leads, the data
    // ZDC/sEPD DSTs of the segment are synced to its events.
    // From DST_CALOFITTING only the raw inputs of the data production are read (GL1 packet, raw MBD),
    // as in the production DST_CALOFITTING. Not read: its data-only towers (same node paths as the
    // embedded towers of DST_CALO) and its reco nodes (made with an older tag, e.g. MinimumBiasInfov1
    // cannot hold the bunch class, so no t0 and no centrality).
    Fun4AllInputManager* in_calofit = new Fun4AllDstInputManager("calofitting");
    in_calofit->BranchSelect("*", 0);
    in_calofit->BranchSelect("DST#Sync", 1);
    in_calofit->BranchSelect("DST#EventHeader", 1);
    in_calofit->BranchSelect("DST#PacketsKeep#14001", 1);
    in_calofit->BranchSelect("DST#MBD#MbdRawContainer", 1);
    in_calofit->AddFile(flist_dst_calofit);
    se->registerInputManager(in_calofit);

    const std::vector<std::pair<std::string, std::string>> data_files = {
        {"zdc", flist_dst_zdc},
        {"sepd", flist_dst_sepd}};

    for (const auto& [name, filepath] : data_files)
    {
      Fun4AllInputManager* in = new Fun4AllDstInputManager(name);
      in->AddFile(filepath);
      se->registerInputManager(in);
    }

    // MC side: the overlay DSTs carry the MC run number (21) in their Sync object, so they cannot be
    // synced to the data. Their entries are event aligned with DST_CALOFITTING, so they are read in
    // file order. The embedded towers go to the data top node; the truth jets (with their own
    // Sync/EventHeader) to the MC top node, where TruthJetValidation checks the alignment.
    Fun4AllInputManager* in_calo = new Fun4AllNoSyncDstInputManager("calo");
    in_calo->BranchSelect("DST#Sync", 0);
    in_calo->BranchSelect("DST#EventHeader", 0);
    in_calo->AddFile(overlay_dst_calo);
    se->registerInputManager(in_calo);

    Fun4AllInputManager* in_truthjet = new Fun4AllNoSyncDstInputManager("truthjet", "DST", "TOPMC");
    in_truthjet->AddFile(overlay_dst_truth_jet);
    se->registerInputManager(in_truthjet);
  }
  else if (is_single_dst)
  {
    Fun4AllInputManager* in = new Fun4AllDstInputManager("DSTin");
    if (isFileList)
    {
      in->AddListFile(flist_dst_calofit);
    }
    else
    {
      in->AddFile(flist_dst_calofit);
    }
    se->registerInputManager(in);
  }
  else
  {
    const std::vector<std::pair<std::string, std::string>> input_files = {
        {"calofitting", flist_dst_calofit},
        {"zdc", flist_dst_zdc},
        {"sepd", flist_dst_sepd}};

    for (const auto& [name, filepath] : input_files)
    {
      Fun4AllInputManager* in = new Fun4AllDstInputManager(name);
      if (isFileList)
      {
        in->AddListFile(filepath);
      }
      else
      {
        in->AddFile(filepath);
      }
      se->registerInputManager(in);
    }
  }

  se->run(nEvents+nSkip);
  se->End();

  if (do_detailed)
  {
    se->dumpHistos(output_tree);
  }
  else
  {
    se->dumpHistos(output);
  }

  CDBInterface::instance()->Print();  // print used DB files
  se->PrintTimer();
  delete se;
  std::cout << "All done!" << std::endl;
  gSystem->Exit(0);
  std::quick_exit(0);
}

// ----------------------------------------------------------------------------
// Overloaded Wrapper for Single DST Input
// ----------------------------------------------------------------------------
void Fun4All_BkgSub(const std::string &input_dst,
                    const std::string &input_QVecCalib,
                    const std::string &output,
                    const std::string &output_tree,
                    int do_flow,
                    int nEvents,
                    int nSkip,
                    int event_id,
                    const std::string &dbtag,
                    const std::string &eta_calib_direct_path,
                    const std::string &event_list,
                    bool do_rcone,
                    bool do_mult,
                    const std::string &output_dst,
                    bool do_neg_energy_threshold = true,
                    float neg_energy_threshold = -2.0F,
                    const std::string &zs_calib_cemc = "",
                    const std::string &zs_calib_hcalin = "",
                    const std::string &zs_calib_hcalout = "",
                    bool do_emfrac_calib = false,
                    const std::string &emfrac_calib_path = "",
                    const std::string &overlay_dst_calo = "",
                    const std::string &overlay_dst_truth_jet = "",
                    float jet_pt_min = 10.0F,
                    float truth_jet_pt_min = 5.0F)
{
  Fun4All_BkgSub(input_dst, /*flist_dst_zdc=*/"", /*flist_dst_sepd=*/"",
                 input_QVecCalib, output, output_tree, do_flow, nEvents, nSkip, event_id,
                 dbtag, eta_calib_direct_path, event_list, do_rcone, do_mult, output_dst,
                 do_neg_energy_threshold, neg_energy_threshold,
                 zs_calib_cemc, zs_calib_hcalin, zs_calib_hcalout,
                 do_emfrac_calib, emfrac_calib_path,
                 overlay_dst_calo, overlay_dst_truth_jet, jet_pt_min, truth_jet_pt_min);
}
