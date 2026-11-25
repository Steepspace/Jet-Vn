# =============================================================================================
# Project Setup
# ====================================================================
cmake_minimum_required(VERSION 3.16)
project(sEPD-Study)

# ====================================================================
# Compiler Settings
# ====================================================================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(
    -Wall -Wextra -pedantic -Wshadow -Wconversion -Wsign-conversion
    -Wnull-dereference -Wformat=2 -Wduplicated-cond -Wuseless-cast
    -Wold-style-cast -Wnoexcept -Woverloaded-virtual
)

# ====================================================================
# Find Dependencies
# ====================================================================
find_package(ROOT REQUIRED)
set(SPHENIX_INSTALL "$ENV{OFFLINE_MAIN}")
set(MY_INSTALL "$ENV{MYINSTALL}")

include_directories(
    "${MY_INSTALL}/include"
    SYSTEM ${ROOT_INCLUDE_DIRS}
    SYSTEM "${SPHENIX_INSTALL}/include"
    SYSTEM "${SPHENIX_INSTALL}/rootmacros"
)
link_directories(
    "${MY_INSTALL}/lib"
    "${SPHENIX_INSTALL}/lib"
)

# ====================================================================
# Define Library Linkages
# ====================================================================
set(SPHENIX_LIBS
    calo_io calo_reco fun4all calotrigger centrality ffamodules mbd epd
    eventplaneinfo zdcinfo globalvertex phool phparameter sphenixnpc cdbobjects
    jetbackground g4jets jetbase particleflow g4detectors_io CaloStatusSkimmer
)
set(MY_LIBS sEPDValidation)

# --- Define a variable for common libraries for standalone macros ---
set(STANDALONE_MACRO_LIBS
    ${ROOT_LIBS}
    ROOT::Core      # For gSystem, etc.
    ROOT::RIO       # For TFile, TTree I/O
    ROOT::Hist      # For TH1, TFitResultPtr,
    ROOT::Graf      # For base graphics classes
    ROOT::HistPainter # For TPaletteAxis
    ROOT::Gpad      # For TCanvas, TPad,
    ROOT::Tree      # For TTree
    ${SPHENIX_LIBS}
    ${MY_LIBS}
)

# ====================================================================
# Define Executable Targets
# ====================================================================
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# --- Fun4All_CaloCheck ---
add_executable(Fun4All_CaloCheck macros/fun4all/Fun4All_CaloCheck.C)
target_link_libraries(Fun4All_CaloCheck PRIVATE
    ROOT::Core # Needs explicit Core for gSystem
    ${ROOT_LIBS}
    ${SPHENIX_LIBS}
    ${MY_LIBS}
)

# --- checkCDB ---
add_executable(checkCDB macros/cdb/checkCDB.C)
target_link_libraries(checkCDB PRIVATE ${STANDALONE_MACRO_LIBS})

# --- checkCDBList ---
add_executable(checkCDBList macros/cdb/checkCDBList.C)
target_link_libraries(checkCDBList PRIVATE ${STANDALONE_MACRO_LIBS})

# --- checkTree ---
add_executable(checkTree macros/utils/checkTree.C)
target_link_libraries(checkTree PRIVATE ${STANDALONE_MACRO_LIBS})

# --- createGlobalBadTowerMap ---
add_executable(createGlobalBadTowerMap macros/cdb/createGlobalBadTowerMap.C)
target_link_libraries(createGlobalBadTowerMap PRIVATE ${STANDALONE_MACRO_LIBS})

# --- Jet Ana v3 ---
add_executable(Jet-Anav3 macros/analysis/Jet-Anav3.C)
target_link_libraries(Jet-Anav3 PRIVATE ${STANDALONE_MACRO_LIBS})

# --- Jet Dijet (clean dijet control sample) ---
add_executable(Jet-Dijet macros/analysis/Jet-Dijet.C)
target_link_libraries(Jet-Dijet PRIVATE ${STANDALONE_MACRO_LIBS})

# --- Jet JES (data+MC overlay) ---
add_executable(Jet-JES macros/analysis/Jet-JES.C)
target_link_libraries(Jet-JES PRIVATE ${STANDALONE_MACRO_LIBS})

# --- sEPD QA ---
add_executable(sEPD-QA macros/analysis/sEPD-QA.C)
target_link_libraries(sEPD-QA PRIVATE ${STANDALONE_MACRO_LIBS})

# ====================================================================
# Optional: Add a Cppcheck Target
# ====================================================================
find_program(CPPCHECK_EXECUTABLE cppcheck)
if(CPPCHECK_EXECUTABLE)
    add_custom_target(cppcheck
        COMMAND ${CPPCHECK_EXECUTABLE} --std=c++20 --enable=all --force
                --suppress=missingIncludeSystem --suppress=missingInclude
                -i macros/deprecated
                "macros"
        COMMENT "Running Cppcheck static analysis..."
    )
endif()
