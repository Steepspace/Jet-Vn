#include "TreeFiller.h"

#include <fun4all/Fun4AllHistoManager.h>
#include <fun4all/Fun4AllReturnCodes.h>
#include <fun4all/Fun4AllServer.h>

#include <phool/phool.h>

#include <TBranch.h>
#include <TDirectory.h>
#include <TFile.h>
#include <TTree.h>

#include <iostream>

TreeFiller *TreeFiller::m_instance = nullptr;
TTree *TreeFiller::m_tree = nullptr;
TFile *TreeFiller::m_file = nullptr;

//____________________________________________________________________________..
TreeFiller::TreeFiller(const std::string &tree_name, const std::string &hm_name)
  : SubsysReco("TreeFiller_" + tree_name)
  , m_tree_name(tree_name)
  , m_hm_name(hm_name)
{
  if (!m_instance)
  {
    m_instance = this;
  }
}

//____________________________________________________________________________..
TreeFiller::TreeFiller(const std::string &name, const std::string &tree_name, const std::string &hm_name)
  : SubsysReco(name)
  , m_tree_name(tree_name)
  , m_hm_name(hm_name)
{
  if (!m_instance)
  {
    m_instance = this;
  }
}

//____________________________________________________________________________..
TTree *TreeFiller::getTree(const std::string &tree_name,
                           const std::string & /*hm_name*/,
                           const std::string &tree_title)
{
  // Modules run Init when they are registered, before the TreeFiller is,
  // so the tree is created here and moved to its output file in TreeFiller::Init
  if (!m_tree)
  {
    m_tree = new TTree(tree_name.c_str(), tree_title.c_str());
  }
  return m_tree;
}

//____________________________________________________________________________..
int TreeFiller::Init(PHCompositeNode * /*topNode*/)
{
  TTree *tree = getTree(m_tree_name);

  // No output file: memory resident tree, written by the HistoManager at the end of the job.
  // Every filled basket stays in memory until then.
  if (m_outfilename.empty())
  {
    Fun4AllServer *se = Fun4AllServer::instance();
    Fun4AllHistoManager *hm = se->getHistoManager(m_hm_name);
    if (!hm)
    {
      hm = new Fun4AllHistoManager(m_hm_name);
      se->registerHistoManager(hm);
    }
    hm->registerHisto(tree);
    return Fun4AllReturnCodes::EVENT_OK;
  }

  // The tree is moved to its output file (no entries yet), so filled baskets are flushed to disk
  TDirectory::TContext restore_dir;
  // same compression as Fun4AllHistoManager
  const int compress = 505;
  m_file = TFile::Open(m_outfilename.c_str(), "RECREATE", "", compress);
  if (!m_file || m_file->IsZombie())
  {
    std::cout << PHWHERE << "TreeFiller: Could not open output file " << m_outfilename << std::endl;
    delete m_file;
    m_file = nullptr;
    return Fun4AllReturnCodes::ABORTRUN;
  }
  tree->SetDirectory(m_file);
  // the branches were made without a file, so they have the default compression
  for (TObject *branch : *tree->GetListOfBranches())
  {
    static_cast<TBranch *>(branch)->SetCompressionSettings(compress);
  }
  return Fun4AllReturnCodes::EVENT_OK;
}

//____________________________________________________________________________..
int TreeFiller::InitRun(PHCompositeNode * /*topNode*/)
{
  if (!m_tree)
  {
    m_tree = getTree(m_tree_name);
  }
  return Fun4AllReturnCodes::EVENT_OK;
}

//____________________________________________________________________________..
int TreeFiller::process_event(PHCompositeNode * /*topNode*/)
{
  if (!m_tree)
  {
    m_tree = getTree(m_tree_name);
  }

  if (m_tree)
  {
    m_tree->Fill();
  }
  else
  {
    std::cout << PHWHERE << "TreeFiller: Could not find or create TTree '"
              << m_tree_name << "' in HistoManager '" << m_hm_name << "'!" << std::endl;
    return Fun4AllReturnCodes::ABORTRUN;
  }

  return Fun4AllReturnCodes::EVENT_OK;
}

//____________________________________________________________________________..
int TreeFiller::End(PHCompositeNode * /*topNode*/)
{
  if (Verbosity() > 0 && m_tree)
  {
    std::cout << "TreeFiller::End - Finished filling tree '" << m_tree_name
              << "' with " << m_tree->GetEntries() << " entries." << std::endl;
  }

  if (m_file)
  {
    std::cout << "TreeFiller::End - Writing root file: " << m_file->GetName() << std::endl;
    // kOverwrite: drop the intermediate tree headers from AutoSave
    m_file->Write("", TObject::kOverwrite);
    // Close deletes the tree
    m_file->Close();
    delete m_file;
    m_file = nullptr;
    m_tree = nullptr;
  }
  return Fun4AllReturnCodes::EVENT_OK;
}
