// model.hpp
//
// The shared physical description of the nucleosome positioning
// Hamiltonian: per-site data energy (eseq) and the nearest-neighbor
// WCA-style linker repulsion table (erep). Both the MCMC sampler
// (NucPosMCMC, mcmc.hpp) and the exact transfer-matrix solver
// (NucPosTM, tm.hpp) hold a NucPosModel member and delegate all
// energy computation to it, so the Hamiltonian is defined in exactly
// one place.

#ifndef MODEL_HPP
#define MODEL_HPP

#include <string>
#include <vector>
#include "dtype.hpp"

class NucPosModel {

public:
  struct Params {
    int nucbp;
    int nbp;
    int llink;
    double elink;
    double mu;
  };

  NucPosModel(int nucbp, int nbp, int llink, double elink, double mu);
  explicit NucPosModel(const Params& params);
  ~NucPosModel();

  void setEnergy(std::string dataFile, double emax);
  void setEnergy(const std::vector<double>& eseq, double emax);
  void resetEnergy();

  int nucbp() const;
  int nbp() const;
  int llink() const;
  double elink() const;
  double mu() const;
  int npos() const;
  const std::vector<double>& eseq() const;

  // Repulsion energy for a pair of nucleosomes separated by gap (the
  // distance between their left-aligned start positions). gap must be
  // at least nucbp (the hard-core exclusion distance); returns 0 for
  // gap >= nucbp+llink, since the WCA potential has no effect beyond
  // that range.
  double erep(int gap) const;

private:
  Params params;
  std::vector<double> siteEnergy;
  std::vector<double> erepTable;

  void buildErep();
};

#endif
