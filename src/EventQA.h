// Tell emacs that this is a C++ source
//  -*- C++ -*-.
#pragma once

#include <fun4all/SubsysReco.h>

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

class TriggerAnalyzer;
class PHCompositeNode;
class TH1;
class TH2;

class EventQA : public SubsysReco
{
 public:
  explicit EventQA(const std::string &name = "EventQA");

  int Init(PHCompositeNode *topNode) override;
  int process_event(PHCompositeNode *topNode) override;
  int ResetEvent(PHCompositeNode *topNode) override;
  int End(PHCompositeNode *topNode) override;

  void set_do_abort(bool b) { m_doAbort = b; }
  void set_do_hist(bool b = true) { m_do_hist = b; }
  void set_do_tree(bool b = true) { m_do_tree = b; }
  void set_cent_max(double cent_max) { m_cuts.m_cent_max = cent_max; }

  bool get_do_tree() const { return m_do_tree; }

 private:
  int process_event_check(PHCompositeNode *topNode);
  int process_centrality(PHCompositeNode *topNode);

  bool m_doAbort{true};
  bool m_do_hist{true};
  bool m_do_tree{true};

  struct HistConfig
  {
    unsigned int m_bins_zvtx{240};
    double m_zvtx_low{-60};
    double m_zvtx_high{60};

    unsigned int m_bins_cent{100};
    double m_cent_low{-0.5};
    double m_cent_high{99.5};
  };

  HistConfig m_hist_config;

  enum class EventType : std::uint8_t
  {
    ALL,
    ZVTX,
    ZVTX60,
    ZVTX10,
    MB_TRIG,
    MB,
    CENT
  };

  enum class MinBiasType : std::uint8_t
  {
    BKG_HIGH,
    SIDE_HIT_LOW,
    ZDC_LOW,
    MBD_HIGH
  };

  std::vector<std::string> m_eventType{"All", "Has Z", "|z| < 60 cm", "|z| < 10 cm", "MB Trig", "MB", "Cent"};
  std::vector<std::string> m_MinBias_Type{"MBD Background", "Hits < 2", "ZDC < 60 GeV", "MBD > 2100"};

  std::unique_ptr<TriggerAnalyzer> m_triggerAnalyzer;

  const int m_trig_12 = 12; // MBD N&S >= 2, vtx < 10 cm
  const int m_trig_14 = 14; // MBD N&S >= 2, vtx < 150 cm

  std::vector<int> m_triggerBits = {m_trig_12, m_trig_14};
  std::vector<std::string> m_triggernames = {"MBD N&S >= 2, vtx < 10 cm",
                                             "MBD N&S >= 2, vtx < 150 cm"};

  // Event Selection Flags
  bool m_pass_MB{false};
  bool m_pass_Zvtx{false};
  bool m_didTrig12Fire{false};
  bool m_didTrig14Fire{false};

  // Cuts
  struct EventCuts
  {
    double m_zvtx_max{10}; // cm
    double m_zvtx_max_v2{60}; // cm
    double m_cent_max{60};
  };

  EventCuts m_cuts;

  std::map<std::string, int> m_ctr;

  struct EventData
  {
    int run{0};
    int event{0};
    double zvtx{9999};
    double centrality{9999};
  };

  EventData m_data;

  // Histograms
  TH1* hEvent{nullptr};
  TH1* hEventTrigger{nullptr};
  TH1* hEventMinBias{nullptr};
  TH1* hVtxZ{nullptr};
  TH1* hVtxZ_MB{nullptr};
  TH1* hZVertex{nullptr};
  TH1* hCentrality{nullptr};
  TH1* hCentralityZ60{nullptr};
  TH1* hCentralityZOuter{nullptr};
  TH2* h2ZVertexCentrality{nullptr};

  // Trigger
  std::vector<TH1*> hZVertexTrig;
  std::vector<TH1*> hCentralityTrig;
  std::vector<TH1*> hCentralityZ60Trig;
  std::vector<TH1*> hCentralityZOuterTrig;
  std::vector<TH2*> h2ZVertexCentralityTrig;
};
