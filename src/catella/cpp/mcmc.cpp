// mcmc.cpp

#include <algorithm>
#include <cmath>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include "dtype.hpp"
#include "mcmc.hpp"
#include "tracker.hpp"

using std::shared_ptr;
using std::string;
using std::vector;

NucPosMCMC::NucPosMCMC(int nucbp, int nbp, int llink, double elink,
		       double mu, ulint seed) :
  model(nucbp, nbp, llink, elink, mu) {
  params.model = NucPosModel::Params{nucbp, nbp, llink, elink, mu};
  params.seed = seed;
  reset();
}

NucPosMCMC::NucPosMCMC(const Params& p) :
  model(p.model), params(p) {
  reset();
}

NucPosMCMC::~NucPosMCMC() {}

void NucPosMCMC::reset() {
  // No nucleosomes on the fiber initially
  nucpos = vector<int>();

  // Set up random generator
  mt = std::mt19937(params.seed);
  randMode = std::uniform_int_distribution<int>(0,2);
  randPos = std::uniform_int_distribution<int>(0,model.npos()-1);
  rand = std::uniform_real_distribution<double>(0.0,1.0);

  // Reset the temperature
  temp = 1.0;

  // Reset energy landscape
  model.resetEnergy();
}

void NucPosMCMC::setEnergy(string dataFile, double emax) {
  model.setEnergy(dataFile, emax);
}

void NucPosMCMC::setEnergy(const vector<double>& eseq, double emax) {
  model.setEnergy(eseq, emax);
}

void NucPosMCMC::update() {
  int mode = randMode(mt);
  double p = rand(mt);
  int nnuc = static_cast<int>(nucpos.size());
  int nucbp = model.nucbp();
  int npos = model.npos();
  const vector<double>& eseq = model.eseq();
  if (mode == 0 && nnuc > 0) { // Shift a nuclosome
    // Pick a nucleosome
    std::uniform_int_distribution<int> randNuc(0,nnuc-1);
    int inuc = randNuc(mt);
    int pos = nucpos[inuc];

    // Pick a direction to move (-1 = left, 1 = right)
    std::uniform_int_distribution<int> randDir(0,1);
    int dir = randDir(mt)*2-1;
    int nxt = pos+dir;

    // Skip if shifting the nucleosome causes it to move off the fiber
    if (nxt < 0 || nxt >= npos) return;

    // Get the position of the nearest left/right nucleosomes
    int idown = (inuc-1 >= 0 ? inuc-1 : -1);
    int iup = (inuc+1 < nnuc ? inuc+1 : -1);
    double dErep = 0.0;
    if (iup != -1) {
      int dpos = nucpos[iup]-pos;
      int ndpos = dpos-dir;
      if (ndpos < nucbp) return; // Nucleosomes cannot overlap
      dErep += model.erep(ndpos);
      dErep -= model.erep(dpos);
    }
    if (idown != -1) {
      int dpos = pos-nucpos[idown];
      int ndpos = dpos+dir;
      if (ndpos < nucbp) return; // Nucleosomes cannot overlap
      dErep += model.erep(ndpos);
      dErep -= model.erep(dpos);
    }
    double dEseq = eseq[nxt]-eseq[pos];
    if (p < std::min(1.0, exp((-dEseq-dErep)/temp))) {
	nucpos[inuc] = nxt;
    }
  } else if (mode == 1) { // Add a nucleosome
    // Pick a location to add a nucleosome
    int pos = randPos(mt);
    // Check for nearby nucleosomes
    auto itup = std::lower_bound(nucpos.begin(), nucpos.end(), pos);
    int iup = (itup != nucpos.end()) ?
      std::distance(nucpos.begin(), itup) : -1;
    int idown = (itup == nucpos.begin()) ? -1 :
      std::distance(nucpos.begin(), std::prev(itup));
    double dErep = 0.0;
    if (iup != -1) {
      int dpos = nucpos[iup]-pos;
      if (dpos < nucbp) return; // Nucleosomes cannot overlap
      dErep += model.erep(dpos);
    }
    if (idown != -1) {
      int dpos = pos-nucpos[idown];
      if (dpos < nucbp) return; // Nucleosomes cannot overlap
      dErep += model.erep(dpos);
    }
    if (p < std::min(1.0, static_cast<double>(npos)/(nnuc+1.0)*
		     exp((model.mu()-eseq[pos]-dErep)/temp))) {
      nucpos.insert(itup, pos);
    }
  } else if (mode == 2 && nnuc > 0) { // Remove a nucleosome
    // Pick a nucleosome
    std::uniform_int_distribution<int> randNuc(0,nnuc-1);
    int inuc = randNuc(mt);
    int pos = nucpos[inuc];

    // Compute change in energy from nearest neighbour interactions
    int idown = (inuc-1 >= 0 ? inuc-1 : -1);
    int iup = (inuc+1 < nnuc ? inuc+1 : -1);
    double dErep = 0.0;
    if (iup != -1) {
      int dpos = nucpos[iup]-pos;
      dErep -= model.erep(dpos);
    }
    if (idown != -1) {
      int dpos = pos-nucpos[idown];
      dErep -= model.erep(dpos);
    }
    if (p < std::min(1.0, nnuc/static_cast<double>(npos)*
		     exp((-model.mu()+eseq[pos]-dErep)/temp))) {
      nucpos.erase(nucpos.begin()+inuc);
    }
  }
}

void NucPosMCMC::run(lint nsweep, double startTemp, double endTemp,
		     Cooling coolOption) {
  if (startTemp <= 0)
    throw std::runtime_error("startTemp must be positive");
  temp = startTemp;
  for (auto& t : trackers) t->initialize(0, *this);
  output(0); // Output the frame without any nucleosome
  for (int n = 0; n < nsweep; n++) {
    // Update temperature
    double progress = (nsweep > 1) ? static_cast<double>(n)/(nsweep-1) : 1.0;
    switch (coolOption) {
    case Cooling::Linear:
      temp = startTemp + progress * (endTemp-startTemp); break;
    case Cooling::Geometric:
      temp = startTemp * pow(endTemp/startTemp, progress); break;
    case Cooling::Constant:
      break; // Do nothing
    }
    // nsweep = 3*npos as there are 3 modes for changing the system (adding,
    // removing, and shifting a nucleosome)
    for (int i = 0; i < 3*model.npos(); i++) {
      update();
    }
    output(n+1);
  }
  for (auto& t : trackers) t->finalize(nsweep, *this);
}

void NucPosMCMC::output(lint time) {
  for (auto& t : trackers) {
    t->track(time, *this);
  }
}

const vector<int>& NucPosMCMC::getNucPos() const {
  return nucpos;
}

double NucPosMCMC::getEnergy() const {
  double totalEseq = 0.0;
  double totalErep = 0.0;
  const vector<double>& eseq = model.eseq();
  for (size_t i = 0; i < nucpos.size(); i++) {
    int pos = nucpos[i];
    totalEseq += eseq[pos];
    if (i > 0) {
      int dpos = nucpos[i]-nucpos[i-1];
      if (dpos >= model.nucbp()) totalErep += model.erep(dpos);
    }
  }
  return totalEseq + totalErep - model.mu()*nucpos.size();
}

double NucPosMCMC::getTemp() const {
  return temp;
}

const vector<double>& NucPosMCMC::getSeqEnergy() const {
  return model.eseq();
}

const NucPosMCMC::Params& NucPosMCMC::getParams() const {
  return params;
}

void NucPosMCMC::addTracker(shared_ptr<Tracker> tracker) {
  trackers.push_back(tracker);
}
