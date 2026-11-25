#ifndef PROVENANCE_H
#define PROVENANCE_H

#include <metadata/MetaDataTree.h>

#include <TFile.h>
#include <TTree.h>

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

R__LOAD_LIBRARY(libmetadata.so)

namespace Provenance
{
  // Appends the job provenance tree "provenance" (MetaDataTree::Make: CDB global tag, timestamp,
  // the CDB payloads used, the input files) to each output file, one entry per job (hadd appends
  // the entries). Check with check_provenance / cdb-frozen-tag-check / extract-cdb-info.
  //
  // Call after se->End() and after the files are written and closed (dumpHistos, TreeFiller,
  // module End, ...), but before delete se: Make reads the input managers of the server and the
  // payloads of CDBInterface. Files that do not exist are skipped. Returns the number of files written.
  inline int Write(const std::vector<std::string> &files)
  {
    std::vector<std::string> unique_files;
    for (const auto &file : files)
    {
      if (!file.empty() && file != "none" &&
          std::find(unique_files.begin(), unique_files.end(), file) == unique_files.end())
      {
        unique_files.push_back(file);
      }
    }

    int nwritten = 0;
    for (const auto &file : unique_files)
    {
      if (!std::filesystem::exists(file))
      {
        std::cout << "Provenance: " << file << " does not exist, no provenance tree written" << std::endl;
        continue;
      }

      TFile f(file.c_str(), "UPDATE");
      if (f.IsZombie())
      {
        std::cout << "Provenance: could not open " << file << ", no provenance tree written" << std::endl;
        continue;
      }

      // owned by f, written with the compression of the file
      TTree *prov = MetaDataTree::Make(&f);
      if (!prov)
      {
        std::cout << "Provenance: could not make the provenance tree for " << file << std::endl;
        continue;
      }
      prov->Write();
      f.Close();
      std::cout << "Provenance: wrote provenance tree to " << file << std::endl;
      ++nwritten;
    }
    return nwritten;
  }
}  // namespace Provenance

#endif  // PROVENANCE_H
