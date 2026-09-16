#include "myUtils.C"

// ====================================================================
// sPHENIX Includes
// ====================================================================
#include <calobase/TowerInfoDefs.h>
#include <cdbobjects/CDBTTree.h>

// ====================================================================
// ROOT Includes
// ====================================================================
#include <TFile.h>
#include <TROOT.h>

// ====================================================================
// Standard C++ Includes
// ====================================================================
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

// ====================================================================
// The Analysis Class
// ====================================================================
class CreateGlobalBadTowerMap
{
 public:
  // The constructor takes the configuration
  CreateGlobalBadTowerMap(std::string input_csv, std::string output_cdbttree, float threshold)
    : m_input_csv(std::move(input_csv))
    , m_output_cdbttree(std::move(output_cdbttree))
    , m_threshold(threshold)
  {
  }

  void run()
  {
    makeMap();
  }

 private:
  // Configuration stored as members
  std::filesystem::path m_input_csv;
  std::filesystem::path m_output_cdbttree;
  float m_threshold{0.1F};

  // --- Private Helper Methods ---
  void makeMap();
};

// ====================================================================
// Implementation of the Class Methods
// ====================================================================
void CreateGlobalBadTowerMap::makeMap()
{
  if (!m_output_cdbttree.parent_path().empty())
  {
    std::filesystem::create_directories(m_output_cdbttree.parent_path());
  }

  std::unordered_set<int> hot_towers;

  bool success = myUtils::readCSV(m_input_csv, [&](const std::string& line) {
    std::vector<std::string> cells = myUtils::split(line, ',');
    // Expected CSV format: TowerIndex,ieta,iphi,TowerKey,HotRunCount,HotRunFraction
    if (cells.size() >= 6)
    {
      int tower_key = std::stoi(cells[3]);
      float hot_fraction = std::stof(cells[5]);
      if (hot_fraction > m_threshold)
      {
        hot_towers.insert(tower_key);
      }
    }
  });

  if (!success)
  {
    throw std::runtime_error(std::format("Failed to read CSV file: {}", m_input_csv.string()));
  }

  std::unique_ptr<CDBTTree> cdbttree_output = std::make_unique<CDBTTree>(m_output_cdbttree);

  constexpr unsigned int ntowers = 24576; // CEMC towers (256 phi x 96 eta)

  for (unsigned int channel = 0; channel < ntowers; ++channel)
  {
    int key = static_cast<int>(TowerInfoDefs::encode_emcal(channel));
    int status = hot_towers.contains(key) ? 2 : 0;

    cdbttree_output->SetIntValue(key, "status", status);
  }

  cdbttree_output->Commit();
  cdbttree_output->WriteCDBTTree();
}

int main(int argc, const char *const argv[])
{
  gROOT->SetBatch(true);

  if (argc < 2 || argc > 4)
  {
    std::cout << "Usage: " << argv[0] << " <input_csv> [output_cdbttree] [threshold]" << std::endl;
    return 1;
  }

  std::string input_csv = argv[1];
  std::string output_cdbttree = (argc >= 3) ? argv[2] : "CEMC_GlobalBadTowerMap.root";
  float threshold = (argc >= 4) ? std::stof(argv[3]) : 0.1F;

  try
  {
    CreateGlobalBadTowerMap analysis(input_csv, output_cdbttree, threshold);
    analysis.run();
  }
  catch (const std::exception &e)
  {
    std::cout << std::format("An exception occurred: {}", e.what()) << std::endl;
    return 1;
  }

  std::cout << "Analysis complete." << std::endl;
  return 0;
}
