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
  ~EventQA() override = default;

  // -- Rule of Five: explicitly delete copy and move semantics --
  EventQA(const EventQA &) = delete;
  EventQA &operator=(const EventQA &) = delete;
  EventQA(EventQA &&) = delete;
  EventQA &operator=(EventQA &&) = delete;

  int Init(PHCompositeNode *topNode) override;
  int process_event(PHCompositeNode *topNode) override;
  int ResetEvent(PHCompositeNode *topNode) override;
  int End(PHCompositeNode *topNode) override;

  void set_do_abort(bool b) { m_doAbort = b; }
  void set_do_hist(bool b = true) { m_do_hist = b; }
  void set_do_tree(bool b = true) { m_do_tree = b; }
  void set_cent_max(double cent_max) { m_cuts.m_cent_max = cent_max; }
  void set_sigma_mbd(double s) { m_sigma_mbd = s; }
  void set_check_mb(bool b) { m_check_mb = b; }
  void set_check_centrality(bool b) { m_check_centrality = b; }
  void set_strict_node_check(bool b) { m_strict_node_check = b; }

  bool get_do_tree() const { return m_do_tree; }

 private:
  int process_event_check(PHCompositeNode *topNode);
  int process_centrality(PHCompositeNode *topNode);

  bool m_doAbort{true};
  bool m_do_hist{true};
  bool m_do_tree{true};
  bool m_check_mb{true};
  bool m_check_centrality{true};
  bool m_strict_node_check{false};

  double m_sigma_mbd{6.324};  // MBD cross section [barns]

  int m_total_events{0};

  // Luminosity / Trigger Event Counters (for |z| < 10 cm)
  int m_n_zvtx10{0};
  int m_n_zvtx10_trig_or{0};
  int m_n_zvtx10_trig12{0};
  int m_n_zvtx10_trig14{0};

  struct HistConfig
  {
    unsigned int m_bins_zvtx{600};
    double m_zvtx_low{-150};
    double m_zvtx_high{150};

    unsigned int m_bins_cent{100};
    double m_cent_low{-0.5};
    double m_cent_high{99.5};
  };

  HistConfig m_hist_config;

  enum class EventType : std::uint8_t
  {
    ALL,
    ZVTX,
    ZVTX150,
    ZVTX10,
    MB_TRIG,
    MB,
    CENT
  };

  enum class EventTriggerType : std::uint8_t
  {
    ZVTX10,
    TRIG12,
    TRIG14,
    TRIG12_OR_TRIG14,
    ZVTX10_TRIG12,
    ZVTX10_TRIG14,
    ZVTX10_TRIG12_OR_TRIG14
  };

  enum class MinBiasType : std::uint8_t
  {
    BKG_HIGH,
    SIDE_HIT_LOW,
    ZDC_LOW,
    MBD_HIGH
  };

  std::vector<std::string> m_eventType{"All", "Has Z", "|z| < 150 cm", "|z| < 10 cm", "MB Trig", "MB", "Cent"};
  std::vector<std::string> m_eventTriggerType{
      "|z| < 10 cm",
      "Trig 12",
      "Trig 14",
      "Trig 12 | Trig 14",
      "|z| < 10 cm & Trig 12",
      "|z| < 10 cm & Trig 14",
      "|z| < 10 cm & (Trig 12 | Trig 14)"};
  std::vector<std::string> m_MinBias_Type{"MBD Background", "Hits < 2", "ZDC < 60 GeV", "MBD > 2100"};

  std::unique_ptr<TriggerAnalyzer> m_triggerAnalyzer;

  const int m_trig_12 = 12; // MBD N&S >= 2, vtx < 10 cm
  const int m_trig_14 = 14; // MBD N&S >= 2, vtx < 150 cm

  std::vector<int> m_triggerBits = {m_trig_12, m_trig_14};
  std::vector<std::string> m_triggernames = {"MBD N&S >= 2, vtx < 10 cm",
                                             "MBD N&S >= 2, vtx < 150 cm"};

  enum TrigIdx : std::size_t
  {
    TRIG12 = 0,
    TRIG12_MB = 1,
    TRIG14 = 2,
    TRIG14_MB = 3,
    NUM_TRIG = 4
  };

  // Event Selection Flags
  bool m_pass_MB{false};
  bool m_pass_Zvtx{false};
  bool m_didTrig12Fire{false};
  bool m_didTrig14Fire{false};

  // Cuts
  struct EventCuts
  {
    double m_zvtx_max{10}; // cm
    double m_zvtx_max_v2{150}; // cm
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
  TH1* hZVertex_Trig12_or_Trig14{nullptr};
  TH1* hZVertex_Trig12_or_Trig14_MB{nullptr};
  TH1* hLuminosity{nullptr};
  TH1* hCentrality{nullptr};
  TH1* hCentralityZ150{nullptr};
  TH1* hCentralityZOuter{nullptr};
  TH2* h2ZVertexCentrality{nullptr};

  // Trigger (paired by trigger bit: [relaxed (trigger-only), tight (trigger + MB)])
  // e.g. [Trig12, Trig12_MB, Trig14, Trig14_MB]
  std::vector<TH1*> hZVertexTrig;
  std::vector<TH1*> hCentralityTrig;
  std::vector<TH1*> hCentralityZ150Trig;
  std::vector<TH1*> hCentralityZOuterTrig;
  std::vector<TH2*> h2ZVertexCentralityTrig;
};
