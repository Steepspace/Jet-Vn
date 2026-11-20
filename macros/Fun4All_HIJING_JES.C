// c++ includes --
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// root includes --
#include <TSystem.h>
#include <TROOT.h>

#include <mbd/MbdReco.h>
#include <centrality/CentralityReco.h>
#include <calotrigger/MinimumBiasClassifier.h>

#include <ffamodules/CDBInterface.h>
#include <ffamodules/FlagHandler.h>

#include <fun4all/Fun4AllBase.h>
#include <fun4all/Fun4AllDstInputManager.h>
#include <fun4all/Fun4AllInputManager.h>
#include <fun4all/Fun4AllServer.h>
#include <fun4all/Fun4AllUtils.h>

#include <phool/recoConsts.h>

#include <sepdvalidation/CaloQA.h>
#include <sepdvalidation/EventQA.h>
#include <sepdvalidation/EventSkip.h>
#include <sepdvalidation/GlobalQA.h>
#include <sepdvalidation/JetValidationv3.h>
#include <sepdvalidation/TruthJetValidation.h>

#include <treefiller/TreeFiller.h>

#include "Calo_Calib.C"
#include "HIJetReco.C"

R__LOAD_LIBRARY(libg4detectors_io.so)
R__LOAD_LIBRARY(libphfield_io.so)
R__LOAD_LIBRARY(libphgeom_io.so)
// dictionaries of the tracker geometries in the DST RUN node (not used, only silences the read errors)
R__LOAD_LIBRARY(libmvtx_io.so)
R__LOAD_LIBRARY(libintt_io.so)
R__LOAD_LIBRARY(libmicromegas_io.so)
R__LOAD_LIBRARY(libffamodules.so)
R__LOAD_LIBRARY(libmbd.so)
R__LOAD_LIBRARY(libsEPDValidation.so)
R__LOAD_LIBRARY(libTreeFiller.so)

// PYTHIA8 jets embedded in HIJING (MDC2 run 31, sHijing 0-20 fm, no pileup) JES:
// iterative jets without flow modulation (do_flow = 0, so the "iter" chain is no-flow), R = 0.2 and 0.3,
// matched offline to the (PYTHIA-only) truth jets.
// Inputs are the aligned DST_CALO_CLUSTER, DST_GLOBAL and DST_TRUTH_JET of the same segments (single files or lists).
// The calibrated towers are rebuilt from the raw towers with Process_Calo_Calib (the DST TOWERINFO_CALIB_* are not read).
// DST_GLOBAL holds the MBD PMTs (MbdPmtContainer) and the MBD vertex (GlobalVertexMap), but no MbdOut, which
// CentralityReco needs (its MBD z picks the vertex scale). MbdReco rebuilds MbdOut from the PMTs (simulation path:
// no raw data, so only MbdEvent::Calculate runs); it is identical to the MbdOut of the production DST_MBD_EPD.
// No vertex reco is run: the DST GlobalVertexMap is used.
// Event selection as in data, minus the trigger: |z| < 10 cm, minimum bias (simulation mode of MinimumBiasClassifier)
// and centrality < cent_max. MDC2 has no centrality scale / vertex scale payloads, so all three centrality
// calibrations are read from the given files (the HIJING ones by default).
void Fun4All_HIJING_JES(const std::string &flist_calo = "DST_CALO_CLUSTER_pythia8_Jet20_sHijing_0_20fm-0000000031-000000.root",
                        const std::string &flist_global = "DST_GLOBAL_pythia8_Jet20_sHijing_0_20fm-0000000031-000000.root",
                        const std::string &flist_truth_jet = "DST_TRUTH_JET_pythia8_Jet20_sHijing_0_20fm-0000000031-000000.root",
                        const std::string &output = "test.root",
                        const std::string &output_tree = "tree.root",
                        int nEvents = 100,
                        int nSkip = 0,
                        const std::string &dbtag = "MDC2",
                        float jet_pt_min = 10.0F,
                        float truth_jet_pt_min = 5.0F,
                        float cent_max = 80.0F,
                        const std::string &cent_scale = "/sphenix/user/dlis/Projects/centrality/cdb/calibrations/scales/cdb_centrality_scale_1.root",
                        const std::string &cent_vtx_scale = "/sphenix/user/dlis/Projects/centrality/cdb/calibrations/vertexscales/cdb_centrality_vertex_scale_1.root",
                        const std::string &cent_divs = "/sphenix/user/dlis/Projects/centrality/cdb/calibrations/divs/cdb_centrality_1.root")
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

  for (const std::string &path : {cent_scale, cent_vtx_scale, cent_divs})
  {
    if (gSystem->AccessPathName(path.c_str()))
    {
      std::cout << "Error: centrality calibration file does not exist: " << path << std::endl;
      return;
    }
  }

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
  std::cout << "cent_max: " << cent_max << std::endl;
  std::cout << "cent_scale: " << cent_scale << std::endl;
  std::cout << "cent_vtx_scale: " << cent_vtx_scale << std::endl;
  std::cout << "cent_divs: " << cent_divs << std::endl;
  std::cout << "########################" << std::endl;

  Fun4AllServer *se = Fun4AllServer::instance();
  se->Verbosity(Fun4AllBase::VERBOSITY_SOME);
  se->VerbosityDownscale(100);

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

  // Calo Calib: also registers the calorimeter geometry (TOWERGEOM_*) for the retowering and the jet validation
  Process_Calo_Calib();

  // Truth Jets: counts the analyzed events for the luminosity, so it runs before any event selection
  TruthJetValidation* truth_jet_validation = new TruthJetValidation();
  truth_jet_validation->set_jet_radii({0.2F, 0.3F});
  truth_jet_validation->set_jet_pt_min(truth_jet_pt_min);
  truth_jet_validation->Verbosity(Fun4AllBase::VERBOSITY_QUIET);
  se->registerSubsystem(truth_jet_validation);

  // MBD Reco: MbdOut (and MbdVertexMap) from the DST MbdPmtContainer, for CentralityReco
  MbdReco* mbdreco = new MbdReco();
  se->registerSubsystem(mbdreco);

  // Minimum Bias Classifier, simulation mode: no ZDC, no bunch classes (MBD PMTs and MBD vertex from DST_GLOBAL)
  // It does not abort events: EventQA applies the MB decision, so its histograms see the failures
  MinimumBiasClassifier* mb = new MinimumBiasClassifier();
  mb->setIsSim(true);
  mb->setOverwriteScale(cent_scale);
  mb->setOverwriteVtx(cent_vtx_scale);
  mb->Verbosity(Fun4AllBase::VERBOSITY_QUIET);
  se->registerSubsystem(mb);

  // Centrality Reco
  CentralityReco* cent = new CentralityReco();
  cent->setOverwriteScale(cent_scale);
  cent->setOverwriteVtx(cent_vtx_scale);
  cent->setOverwriteDivs(cent_divs);
  se->registerSubsystem(cent);

  // Event QA: no trigger (no GL1 in simulation); MB, |z| < 10 cm and centrality < cent_max as in data
  // strict: abort the run if MinimumBiasInfo / MinBiasParams / CentralityInfo are missing instead of skipping the cuts
  EventQA* event_qa = new EventQA();
  event_qa->set_check_trigger(false);
  event_qa->set_check_mb(true);
  event_qa->set_check_centrality(true);
  event_qa->set_strict_node_check(true);
  event_qa->set_zvtx_max(10);
  event_qa->set_cent_max(cent_max);
  event_qa->Verbosity(Fun4AllBase::VERBOSITY_QUIET);
  se->registerSubsystem(event_qa);

  // Jet Reco
  // with do_flow == 0 the default iterative chain is already no-flow
  HIJETS::do_flow = 0;
  Enable::HIJETS_TOWER = true;
  Enable::HIJETS_TOWER_NOBKG = false;
  Enable::HIJETS_TOWER_NOFLOW = false;
  Enable::HIJETS_TOWER_MULTSUB = false;
  Enable::HIJETS_TOWER_MULTSUB_NOFLOW = false;
  Enable::HIJETS_TOWER_CALIB_EMFRAC = false;
  HIJetReco();

  // Global QA: MBD charge per side (from MinBiasParams), no event plane / sEPD
  // registered before CaloQA, which reads the MBD charge of the current event from it
  GlobalQA* global_qa = new GlobalQA();
  global_qa->set_do_ep(false);
  global_qa->set_do_sepd(false);
  global_qa->set_do_mbd(true);
  global_qa->Verbosity(Fun4AllBase::VERBOSITY_QUIET);
  se->registerSubsystem(global_qa);

  // Calo QA: total EMCal / IHCal / OHCal energy; the histograms include h2TotalCaloE_MBD for a HIJING Calo-MBD band
  CaloQA* calo_qa = new CaloQA();
  calo_qa->set_do_mbd(true);
  calo_qa->set_do_sepd(false);
  calo_qa->Verbosity(Fun4AllBase::VERBOSITY_QUIET);
  se->registerSubsystem(calo_qa);

  // Jet Validation: iter only (no flow)
  JetValidationv3* jet_validation = new JetValidationv3();
  jet_validation->set_do_unsub(false);
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

  // the waveforms and clusters are not used, and the calibrated towers are rebuilt by Process_Calo_Calib
  Fun4AllInputManager* in_calo = new Fun4AllDstInputManager("calo");
  for (const std::string det : {"CEMC", "HCALIN", "HCALOUT"})
  {
    in_calo->BranchSelect("DST#" + det + "#WAVEFORM_" + det, 0);
    in_calo->BranchSelect("DST#" + det + "#TOWERINFO_CALIB_" + det, 0);
  }
  in_calo->BranchSelect("DST#CEMC#CLUSTERINFO_CEMC", 0);

  // the sEPD towers are not used (no event plane)
  Fun4AllInputManager* in_global = new Fun4AllDstInputManager("global");
  in_global->BranchSelect("DST#EPD#TOWERINFO_SIM_EPD", 0);
  in_global->BranchSelect("DST#EPD#TOWERINFO_CALIB_EPD", 0);

  const std::vector<std::pair<Fun4AllInputManager*, std::string>> input_files = {
      {in_calo, flist_calo},
      {in_global, flist_global},
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
