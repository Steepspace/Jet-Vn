// c++ includes --
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

#include <ffamodules/CDBInterface.h>
#include <ffamodules/FlagHandler.h>

#include <fun4all/Fun4AllDstInputManager.h>
#include <fun4all/Fun4AllDstOutputManager.h>
#include <fun4all/Fun4AllInputManager.h>
#include <fun4all/Fun4AllServer.h>
#include <fun4all/Fun4AllBase.h>
#include <fun4all/Fun4AllUtils.h>

#include <caloreco/CaloTowerBuilder.h>
#include <caloreco/CaloWaveformProcessing.h>

#include <phool/recoConsts.h>

R__LOAD_LIBRARY(libcalo_reco.so)
R__LOAD_LIBRARY(libcalotrigger_io.so)

void Fun4All_sEPD_Calib(const std::string &infile_dst_sepd="DST_SEPD_RAW_run3auau_pro001_pcdb001_v001-00068144-00000.root",
                        const std::filesystem::path& outDir = ".",
                        int nEvents = 100,
                        const std::string &dbtag = "newcdbtag",
                        int logInterval = 20)
{
  std::pair<int, int> runseg = Fun4AllUtils::GetRunSegment(infile_dst_sepd);
  int runnumber = runseg.first;
  int segment = runseg.second;

  std::cout << "########################" << std::endl;
  std::cout << "Run Parameters" << std::endl;
  std::cout << "input sepd: " << infile_dst_sepd << std::endl;
  std::cout << "Run: " << runnumber << std::endl;
  std::cout << "Segment: " << segment << std::endl;
  std::cout << "Events: " << nEvents << std::endl;
  std::cout << "dbtag: " << dbtag << std::endl;
  std::cout << "Log Interval: " << logInterval << std::endl;
  std::cout << "########################" << std::endl;

  Fun4AllServer *se = Fun4AllServer::instance();
  se->Verbosity(Fun4AllBase::VERBOSITY_SOME);
  se->VerbosityDownscale(logInterval);

  recoConsts *rc = recoConsts::instance();

  // conditions DB flags and timestamp
  rc->set_StringFlag("CDB_GLOBALTAG", dbtag);
  rc->set_uint64Flag("TIMESTAMP", runnumber);
  CDBInterface::instance()->Verbosity(Fun4AllBase::VERBOSITY_SOME);

  FlagHandler* flag = new FlagHandler();
  se->registerSubsystem(flag);

  CaloTowerDefs::BuilderType buildertype = CaloTowerDefs::kPRDFTowerv4;

  // sEPD Reconstruction--Calib Info: Packets -> TOWERS_SEPD
  CaloTowerBuilder* caEPD = new CaloTowerBuilder("SEPDBUILDER");
  caEPD->set_detector_type(CaloTowerDefs::SEPD);
  caEPD->set_builder_type(buildertype);
  caEPD->set_processing_type(CaloWaveformProcessing::TEMPLATE);
  caEPD->set_nsamples(12);
  caEPD->set_offlineflag();
  se->registerSubsystem(caEPD);

  std::filesystem::path p(infile_dst_sepd);
  std::string outFileName = p.filename().string();

  size_t start_pos = outFileName.find("RAW");
  if (start_pos != std::string::npos)
  {
    outFileName.replace(start_pos, 3, "CALIB");
  }
  else
  {
    // Optional fallback just in case the input file doesn't have "RAW" in the name
    outFileName = std::format("DST_SEPD_CALIB-{:08d}-{:05d}.root", runnumber, segment);
  }

  std::filesystem::path outDirPath = outDir / std::to_string(runnumber);
  std::filesystem::path outFilePathName = outDirPath / outFileName;

  std::filesystem::create_directories(outDirPath);

  Fun4AllDstOutputManager* out = new Fun4AllDstOutputManager("DSTOUT", outFilePathName.string());
  out->AddNode("Sync");
  out->AddNode("EventHeader");
  out->AddNode("TOWERS_SEPD");
  se->registerOutputManager(out);

  Fun4AllInputManager* In = new Fun4AllDstInputManager("sepd");
  In->AddFile(infile_dst_sepd);
  se->registerInputManager(In);

  se->run(nEvents);
  se->End();

  CDBInterface::instance()->Print();  // print used DB files
  se->PrintTimer();
  delete se;
  std::cout << "Saved calibrated data to: " << outFilePathName.string() << std::endl;
  std::cout << "All done!" << std::endl;
  gSystem->Exit(0);
}
