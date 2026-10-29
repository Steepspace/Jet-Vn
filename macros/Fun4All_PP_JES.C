// c++ includes --
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// root includes --
#include <TSystem.h>
#include <TROOT.h>

#include <ffamodules/CDBInterface.h>
#include <ffamodules/FlagHandler.h>

#include <fun4all/Fun4AllBase.h>
#include <fun4all/Fun4AllDstInputManager.h>
#include <fun4all/Fun4AllInputManager.h>
#include <fun4all/Fun4AllRunNodeInputManager.h>
#include <fun4all/Fun4AllServer.h>
#include <fun4all/Fun4AllUtils.h>

#include <phool/recoConsts.h>

#include <sepdvalidation/EventSkip.h>
#include <sepdvalidation/EventQA.h>
#include <sepdvalidation/JetValidationv3.h>
#include <sepdvalidation/TruthJetValidation.h>

#include <treefiller/TreeFiller.h>

#include "HIJetReco.C"

R__LOAD_LIBRARY(libg4detectors_io.so)
R__LOAD_LIBRARY(libffamodules.so)
R__LOAD_LIBRARY(libsEPDValidation.so)
R__LOAD_LIBRARY(libTreeFiller.so)

// p+p Pythia8 MC JES: unsubtracted and iterative (no flow) jets, R = 0.2 and 0.3, matched offline to the truth jets.
// Inputs are the aligned DST_CALO_CLUSTER, DST_GLOBAL and DST_TRUTH_JET of the same segments (single files or lists).
// The DSTs already hold the calibrated towers, the MBD and the MBD vertex, so no calo calibration or MBD/vertex reco
// is run. The only event selection is a valid MBD vertex with |z| < 10 cm (no trigger, minimum bias or centrality).
void Fun4All_PP_JES(const std::string &flist_calo = "DST_CALO_CLUSTER_pythia8_Jet20-0000000028-000000.root",
                    const std::string &flist_global = "DST_GLOBAL_pythia8_Jet20-0000000028-000000.root",
                    const std::string &flist_truth_jet = "DST_TRUTH_JET_pythia8_Jet20-0000000028-000000.root",
                    const std::string &output = "test.root",
                    const std::string &output_tree = "tree.root",
                    int nEvents = 100,
                    int nSkip = 0,
                    const std::string &dbtag = "MDC2",
                    float jet_pt_min = 10.0F,
                    float truth_jet_pt_min = 5.0F)
{
  // Extract runnumber from first file within list
  bool isFileList = !flist_calo.ends_with(".root");
  std::string first_file = flist_calo;
  if (isFileList)
  {
    std::ifstream infile_stream(flist_calo);
    if (!infile_stream)
    {
      std::cout << "Error: Could not open file list " << flist_calo << std::endl;
      return;
    }
    getline(infile_stream, first_file);
  }
  int runnumber = Fun4AllUtils::GetRunSegment(first_file).first;

  std::cout << "########################" << std::endl;
  std::cout << "Run Parameters" << std::endl;
  std::cout << "input calo: " << flist_calo << std::endl;
  std::cout << "input global: " << flist_global << std::endl;
  std::cout << "input truth jet: " << flist_truth_jet << std::endl;
  std::cout << "Run: " << runnumber << std::endl;
  std::cout << "output: " << output << std::endl;
  std::cout << "output tree: " << output_tree << std::endl;
  std::cout << "nEvents: " << nEvents << std::endl;
  std::cout << "nSkip: " << nSkip << std::endl;
  std::cout << "dbtag: " << dbtag << std::endl;
  std::cout << "jet_pt_min: " << jet_pt_min << std::endl;
  std::cout << "truth_jet_pt_min: " << truth_jet_pt_min << std::endl;
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
  evtSkip->Verbosity(Fun4AllBase::VERBOSITY_QUIET);
  se->registerSubsystem(evtSkip);

  // Truth Jets: counts the analyzed events for the luminosity, so it runs before any event selection
  TruthJetValidation* truth_jet_validation = new TruthJetValidation();
  truth_jet_validation->set_jet_radii({0.2F, 0.3F});
  truth_jet_validation->set_jet_pt_min(truth_jet_pt_min);
  truth_jet_validation->Verbosity(Fun4AllBase::VERBOSITY_QUIET);
  se->registerSubsystem(truth_jet_validation);

  // Event QA: no trigger (no GL1 in simulation), no minimum bias and no centrality, only |z| < 10 cm
  EventQA* event_qa = new EventQA();
  event_qa->set_check_trigger(false);
  event_qa->set_check_mb(false);
  event_qa->set_check_centrality(false);
  event_qa->set_zvtx_max(10);
  event_qa->Verbosity(Fun4AllBase::VERBOSITY_QUIET);
  se->registerSubsystem(event_qa);

  // Jet Reco
  // with do_flow == 0 the default iterative chain is already no-flow
  HIJETS::do_flow = 0;
  Enable::HIJETS_TOWER = true;
  Enable::HIJETS_TOWER_NOBKG = true;
  Enable::HIJETS_TOWER_NOFLOW = false;
  Enable::HIJETS_TOWER_MULTSUB = false;
  Enable::HIJETS_TOWER_MULTSUB_NOFLOW = false;
  Enable::HIJETS_TOWER_CALIB_EMFRAC = false;
  HIJetReco();

  // Jet Validation
  JetValidationv3* jet_validation = new JetValidationv3();
  jet_validation->set_do_unsub(true);
  jet_validation->set_do_iter(true);
  jet_validation->set_do_iter_noflow(false);
  jet_validation->set_do_mult(false);
  jet_validation->set_do_mult_noflow(false);
  jet_validation->set_do_timing(true);
  jet_validation->set_jet_pt_min(jet_pt_min);
  jet_validation->Verbosity(Fun4AllBase::VERBOSITY_QUIET);
  se->registerSubsystem(jet_validation);

  // Tree Filler
  TreeFiller* tree_filler = TreeFiller::instance();
  tree_filler->setOutfileName(output_tree);
  tree_filler->Verbosity(Fun4AllBase::VERBOSITY_QUIET);
  se->registerSubsystem(tree_filler);

  // calorimeter geometry (TOWERGEOM_*) for the retowering and the jet validation
  Fun4AllInputManager *ingeo = new Fun4AllRunNodeInputManager("DST_GEO");
  ingeo->AddFile(CDBInterface::instance()->getUrl("calo_geo"));
  se->registerInputManager(ingeo);

  // the waveforms and clusters are not used
  Fun4AllInputManager* in_calo = new Fun4AllDstInputManager("calo");
  for (const std::string det : {"CEMC", "HCALIN", "HCALOUT"})
  {
    in_calo->BranchSelect("DST#" + det + "#WAVEFORM_" + det, 0);
  }
  in_calo->BranchSelect("DST#CEMC#CLUSTERINFO_CEMC", 0);

  const std::vector<std::pair<Fun4AllInputManager*, std::string>> input_files = {
      {in_calo, flist_calo},
      {new Fun4AllDstInputManager("global"), flist_global},
      {new Fun4AllDstInputManager("truthjet"), flist_truth_jet}};

  for (const auto& [in, filepath] : input_files)
  {
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

  se->run(nEvents+nSkip);
  se->End();

  se->dumpHistos(output);

  CDBInterface::instance()->Print();  // print used DB files
  se->PrintTimer();
  delete se;
  std::cout << "All done!" << std::endl;
  gSystem->Exit(0);
  std::quick_exit(0);
}
