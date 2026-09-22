// model.cpp

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "model.hpp"

using std::ifstream;
using std::string;
using std::stringstream;
using std::vector;

NucPosModel::NucPosModel(int nucbp, int nbp, int llink, double elink,
			 double mu) :
  params{nucbp, nbp, llink, elink, mu} {
  siteEnergy = vector<double>(nbp, 0.0);
  buildErep();
}

NucPosModel::NucPosModel(const Params& p) : params(p) {
  siteEnergy = vector<double>(params.nbp, 0.0);
  buildErep();
}

NucPosModel::~NucPosModel() {}

void NucPosModel::buildErep() {
  // Precompute repulsion strengths for the linker potential
  erepTable = vector<double>(params.llink, 0.0);
  for (int i = 0; i < params.llink; i++) {
    double sr6 = pow(params.llink/(i+1.0), 6.0) - 1.0;
    erepTable[i] = params.elink*sr6*sr6;
  }
}

void NucPosModel::resetEnergy() {
  std::fill(siteEnergy.begin(), siteEnergy.end(), 0.0);
}

void NucPosModel::setEnergy(string dataFile, double emax) {
  // '!(emax >= 0)' (rather than 'emax < 0') also rejects NaN, since any
  // comparison with NaN is false; +infinity passes, disabling clipping.
  if (!(emax >= 0)) {
    throw std::runtime_error(
      "emax must be non-negative (pass inf to disable clipping).");
  }
  siteEnergy = vector<double>(params.nbp, 0.0);
  ifstream reader;
  reader.open(dataFile);
  if (!reader) {
    throw std::runtime_error("Cannot open the file " + dataFile + ".");
  }
  string line;
  stringstream ss;
  int pos;
  double e;
  while (getline(reader, line)) {
    if (line[0] == '#') continue; // Skip comments
    ss.clear();
    ss.str(line);
    ss >> pos >> e;
    if (pos >= 0 && pos < params.nbp) {
      siteEnergy[pos] = std::clamp(e, -emax, emax);
    }
  }
  reader.close();
}

void NucPosModel::setEnergy(const std::vector<double>& eseq_in,
			    double emax) {
  if (!(emax >= 0)) {
    throw std::runtime_error(
      "emax must be non-negative (pass inf to disable clipping).");
  }
  if (static_cast<int>(eseq_in.size()) != params.nbp) {
    throw std::runtime_error("Energy array size does not match "
			     "the size of the simulated fiber.");
  }
  for (int i = 0; i < params.nbp; i++) {
    siteEnergy[i] = std::clamp(eseq_in[i], -emax, emax);
  }
}

int NucPosModel::nucbp() const { return params.nucbp; }
int NucPosModel::nbp() const { return params.nbp; }
int NucPosModel::llink() const { return params.llink; }
double NucPosModel::elink() const { return params.elink; }
double NucPosModel::mu() const { return params.mu; }
int NucPosModel::npos() const { return params.nbp - params.nucbp + 1; }

const vector<double>& NucPosModel::eseq() const { return siteEnergy; }

double NucPosModel::erep(int gap) const {
  int i = gap - params.nucbp;
  if (i < 0) {
    throw std::runtime_error("erep: gap must be at least nucbp.");
  }
  if (i >= params.llink) {
    return 0.0;
  }
  return erepTable[i];
}
