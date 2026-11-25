AUTOMAKE_OPTIONS = foreign

AM_CPPFLAGS = \
  -I$(includedir) \
  -isystem$(OFFLINE_MAIN)/include \
  -isystem$(ROOTSYS)/include

AM_LDFLAGS = \
  -L$(libdir) \
  -L$(OFFLINE_MAIN)/lib \
  -L$(OFFLINE_MAIN)/lib64

pkginclude_HEADERS = \
  geometry_constants.h \
  EventSkip.h \
  CaloCheck.h \
  CaloQA.h \
  JetValidationv3.h \
  TruthJetValidation.h \
  EventQA.h \
  GlobalQA.h \
  JetUtils.h \
  RandomConeMaker.h \
  RandomConeValidation.h

lib_LTLIBRARIES = \
  libsEPDValidation.la

libsEPDValidation_la_SOURCES = \
  EventSkip.cc \
  CaloCheck.cc \
  CaloQA.cc \
  JetValidationv3.cc \
  TruthJetValidation.cc \
  EventQA.cc \
  GlobalQA.cc \
  JetUtils.cc \
  RandomConeMaker.cc \
  RandomConeValidation.cc

libsEPDValidation_la_LIBADD = \
  -lphool \
  -lSubsysReco \
  -lcentrality_io \
  -lfun4all \
  -lffamodules \
  -lglobalvertex_io \
  -lcalotrigger_io \
  -lcalotrigger \
  -leventplaneinfo_io \
  -lepd_io \
  -ljetbase_io \
  -ljetbackground_io \
  -lphparameter_io \
  -lphhepmc_io \
  -lTreeFiller

BUILT_SOURCES = testexternals.cc

noinst_PROGRAMS = \
  testexternals

testexternals_SOURCES = testexternals.cc
testexternals_LDADD   = libsEPDValidation.la

testexternals.cc:
	echo "//*** this is a generated file. Do not commit, do not edit" > $@
	echo "int main()" >> $@
	echo "{" >> $@
	echo "  return 0;" >> $@
	echo "}" >> $@

clean-local:
	rm -f $(BUILT_SOURCES)
