#ifndef MACRO_EVENTPLANERECO_C
#define MACRO_EVENTPLANERECO_C

#include <eventplaneinfo/EventPlaneReco.h>

#include <fun4all/Fun4AllServer.h>

#include <cctype>
#include <filesystem>
#include <iostream>
#include <string>

R__LOAD_LIBRARY(libeventplaneinfo.so)

// ----------------------------------------------------------------------------
//! sEPD event plane reconstruction (EventPlaneReco), shared by Fun4All_BkgSub and
//! Fun4All_EventQA. Needs TOWERINFO_CALIB_SEPD (EpdReco) and CentralityInfo
//! (CentralityReco) to be registered before it.
// ----------------------------------------------------------------------------
namespace EVENTPLANE
{
  std::string node_input = "TOWERINFO_CALIB_SEPD";
  std::string node_output = "EventplaneinfoMap";
}  // namespace EVENTPLANE

// ----------------------------------------------------------------------------
//! Run number of a Q vector calibration file, from the digits after the last '-' of its
//! name (SEPD_EventPlaneCalib-..._v001-68144.root or ...-68144v2.root). -1 if none.
// ----------------------------------------------------------------------------
int EventPlaneCalibRun(const std::string &path)
{
  std::string stem = std::filesystem::path(path).stem().string();
  std::string tail = stem.substr(stem.find_last_of('-') + 1);

  size_t ndigits = 0;
  while (ndigits < tail.size() && std::isdigit(static_cast<unsigned char>(tail[ndigits])))
  {
    ++ndigits;
  }
  return ndigits > 0 ? std::stoi(tail.substr(0, ndigits)) : -1;
}

// ----------------------------------------------------------------------------
//! True if the calibration is "default" (CDB) or empty, or if its run matches runnumber.
// ----------------------------------------------------------------------------
bool CheckEventPlaneCalibRun(const std::string &calib, int runnumber)
{
  if (calib.empty() || calib == "default")
  {
    return true;
  }

  int calib_run = EventPlaneCalibRun(calib);
  if (calib_run != runnumber)
  {
    std::cout << "Error: Q vector calib run " << calib_run << " does not match calofit run "
              << runnumber << ": " << calib << std::endl;
    return false;
  }
  return true;
}

// ----------------------------------------------------------------------------
//! Register EventPlaneReco. calib: direct path to the Q vector calibration file,
//! or "default" / empty to use the CDB.
// ----------------------------------------------------------------------------
void Process_EventPlane_Reco(const std::string &calib = "default")
{
  Fun4AllServer *se = Fun4AllServer::instance();

  EventPlaneReco *epreco = new EventPlaneReco();
  if (!calib.empty() && calib != "default")
  {
    epreco->set_directURL_EventPlaneCalib(calib);
  }
  epreco->set_inputNode(EVENTPLANE::node_input);
  epreco->set_EventPlaneInfoNodeName(EVENTPLANE::node_output);
  se->registerSubsystem(epreco);
}

#endif
