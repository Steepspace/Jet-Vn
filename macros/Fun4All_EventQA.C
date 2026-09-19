// c++ includes --
#include <string>
#include <iostream>
#include <format>

// root includes --
#include <TSystem.h>
#include <TROOT.h>
#include <TF1.h>

#include <ffamodules/CDBInterface.h>
#include <ffamodules/FlagHandler.h>

#include <fun4all/Fun4AllDstInputManager.h>
#include <fun4all/Fun4AllInputManager.h>
#include <fun4all/Fun4AllServer.h>
#include <fun4all/Fun4AllBase.h>
#include <fun4all/Fun4AllUtils.h>
#include <phool/recoConsts.h>

#include <mbd/MbdReco.h>
#include <mbd/MbdEvent.h>
#include <zdcinfo/ZdcReco.h>
#include <globalvertex/GlobalVertexReco.h>
#include <centrality/CentralityReco.h>
#include <calotrigger/MinimumBiasClassifier.h>
#include <calotrigger/TriggerRunInfoReco.h>

#include <calostatusskimmer/CaloStatusSkimmer.h>

#include <sepdvalidation/EventQA.h>

R__LOAD_LIBRARY(libg4detectors_io.so)
R__LOAD_LIBRARY(libCaloStatusSkimmer.so)
R__LOAD_LIBRARY(libsEPDValidation.so)

void Fun4All_EventQA(const std::string &flist_dst_calofit = "DST_CALOFITTING_run3auau_pro001_pcdb001_v001-00068144-00000.root",
                     const std::string &flist_dst_zdc = "/direct/sphenix+tg+tg01/jets/anarde/run3auau/ZDC/68144/DST_ZDC_CALIB_run3auau_pro001_pcdb001_v001-00068144-00000.root",
                     const std::string& output = "test.root",
                     int nEvents = 100,
                     const std::string& dbtag = "newcdbtag")
{
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

  std::cout << "########################" << std::endl;
  std::cout << "Run Parameters" << std::endl;
  std::cout << "input calofit: " << flist_dst_calofit << std::endl;
  std::cout << "input zdc: " << flist_dst_zdc << std::endl;
  std::cout << "output: " << output << std::endl;
  std::cout << "nEvents: " << nEvents << std::endl;
  std::cout << "dbtag: " << dbtag << std::endl;
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

  CaloStatusSkimmer* css = new CaloStatusSkimmer("CaloStatusSkimmer");
  se->registerSubsystem(css);

  // MBD Reconstruction
  MbdReco* mbdreco = new MbdReco();
  se->registerSubsystem(mbdreco);

  // Official vertex storage
  GlobalVertexReco* gvertex = new GlobalVertexReco();
  gvertex->Verbosity(Fun4AllBase::VERBOSITY_QUIET);
  se->registerSubsystem(gvertex);

  // Trigger Info Reco
  TriggerRunInfoReco* trig = new TriggerRunInfoReco();
  trig->Verbosity(1);
  se->registerSubsystem(trig);

  // custom centrality calib
  std::string cent_calib_dir = "/sphenix/user/anarde/sEPD-Study/centrality_calib";
  std::string cent_divs = std::format("{}/divs/cdb_centrality_{}.root", cent_calib_dir, runnumber);
  // DEFAULT use 68144 for now
  std::string cent_scale = std::format("{}/scales/cdb_centrality_scale_68144.root", cent_calib_dir);
  // std::string cent_scale = std::format("{}/scales/cdb_centrality_scale_{}.root", cent_calib_dir, runnumber);
  std::string cent_vtx = std::format("{}/vertexscales/cdb_centrality_vertex_scale_{}.root", cent_calib_dir, runnumber);

  // Minimum Bias Classifier
  MinimumBiasClassifier* mb = new MinimumBiasClassifier();
  mb->setOverwriteScale(cent_scale);
  mb->setOverwriteVtx(cent_vtx);
  se->registerSubsystem(mb);

  // Centrality Reco
  CentralityReco* cent = new CentralityReco();
  cent->setOverwriteDivs(cent_divs);
  cent->setOverwriteScale(cent_scale);
  cent->setOverwriteVtx(cent_vtx);
  se->registerSubsystem(cent);

  // Event QA
  EventQA* event_qa = new EventQA();
  event_qa->Verbosity(Fun4AllBase::VERBOSITY_QUIET);
  event_qa->set_do_tree(false);
  event_qa->set_cent_max(100);
  se->registerSubsystem(event_qa);

  const std::vector<std::pair<std::string, std::string>> input_files = {
      {"calofitting", flist_dst_calofit},
      {"zdc", flist_dst_zdc}};

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

  se->run(nEvents);
  se->End();

  se->dumpHistos(output);

  CDBInterface::instance()->Print();  // print used DB files
  se->PrintTimer();
  delete se;
  std::cout << "All done!" << std::endl;
  gSystem->Exit(0);
  std::quick_exit(0);
}
