// model.cpp

#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <memory>
#include "dtype.hpp"
#include "model.hpp"
#include "tracker.hpp"

using std::cout;
using std::endl;
using std::iostream;
using std::ifstream;
using std::ofstream;
using std::stringstream;
using std::string;
using std::vector;
using std::shared_ptr;

// Helper functions
double min(double a, double b);

NucPosModel::NucPosModel(int _nucbp, int _nbp, int _llink, double _mu,
			 ulint _seed) :
  nucbp(_nucbp), nbp(_nbp), llink(_llink), mu(_mu), seed(_seed) {
  params.nucbp = nucbp;
  params.nbp = nbp;
  params.llink = llink;
  params.mu = mu;
  params.seed = seed;
  // Precompute repulsion strengths - WCA repulsion
  erep = vector<double>(llink, 0.0);
  double sigma = llink/(pow(2.0,(1.0/6.0)));
  for (int i = 0; i < llink; i++) {
    double sr6 = pow(sigma/(i+1.0),6.0);
    erep[i] = 4*(sr6*sr6-sr6+0.25);
  }
  maxNumOfNuc = nbp/nucbp;
  npos = nbp-nucbp;
  emeth = vector<double>(nbp);
  reset();
}

NucPosModel::NucPosModel(const Params& p) :
  params(p), nucbp(p.nucbp), nbp(p.nbp), llink(p.llink), mu(p.mu),
  seed(p.seed) {
  // Precompute repulsion strengths - WCA repulsion
  erep = vector<double>(llink, 0.0);
  double sigma = llink/(pow(2.0,(1.0/6.0)));
  for (int i = 0; i < llink; i++) {
    double sr6 = pow(sigma/(i+1.0),6.0);
    erep[i] = 4*(sr6*sr6-sr6+0.25);
  }
  maxNumOfNuc = nbp/nucbp;
  npos = nbp-nucbp;
  emeth = vector<double>(nbp);
  reset();
}

NucPosModel::~NucPosModel() {}

void NucPosModel::reset() {
  // No nucleosomes on the fiber initially
  nucpos = vector<int>();

  // Set up random generator
  mt = std::mt19937(seed);  
  randMode = std::uniform_int_distribution<int>(0,2);
  randPos = std::uniform_int_distribution<int>(0,npos-1);
  rand = std::uniform_real_distribution<double>(0.0,1.0);

  // Reset the temperature
  temp = 1.0;

  // Reset energy landscape
  std::fill(emeth.begin(), emeth.end(), 0.0);  
}

void NucPosModel::setMethEnergy(string dataFile, double emax) {
  emeth = vector<double>(nbp, 0.0);
  ifstream reader;
  reader.open(dataFile);
  if (!reader) {
    throw std::runtime_error("Cannot open the file " + dataFile);
  }
  string line;
  stringstream ss;
  int pos;
  double score;
  double pmin = exp(-emax);
  double p;  
  while (getline(reader, line)) {
    if (line[0] == '#') continue; // Skip comments
    ss.clear();
    ss.str(line);
    ss >> pos >> score;
    if (pos >= 0 && pos < nbp) {
      p = 1-score;
      emeth[pos] = p < pmin ? emax : -log(p); // E_meth = -kT log(1-M)
    }
  }
  reader.close();
}

void NucPosModel::setMethEnergy(const std::vector<double>& data, double emax) {
  double pmin = exp(-emax);
  double p;
  if (static_cast<int>(data.size()) != nbp) {
    throw std::runtime_error("Methylation data array size does not match "
			     "the size of the simulated fiber");
  }
  for (int i = 0; i < nbp; i++) {
    p = 1-data[i];
    emeth[i] = p < pmin ? emax : -log(p); // E_meth = -kT log(1-M)
  }
}

void NucPosModel::update() {
  int mode = randMode(mt);
  double p = rand(mt);
  int nnuc = static_cast<int>(nucpos.size());
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
      if (ndpos < llink+nucbp) dErep += erep[ndpos-nucbp];
      if (dpos < llink+nucbp) dErep -= erep[dpos-nucbp];
    }
    if (idown != -1) {
      int dpos = pos-nucpos[idown];
      int ndpos = dpos+dir;
      if (ndpos < nucbp) return; // Nucleosomes cannot overlap
      if (ndpos < llink+nucbp) dErep += erep[ndpos-nucbp];
      if (dpos < llink+nucbp) dErep -= erep[dpos-nucbp];
    }
    double dEmeth = emeth[nxt]-emeth[pos];
    if (p < min(1.0, exp((-dEmeth-dErep)/temp))) {
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
      else if (dpos < llink+nucbp) dErep += erep[dpos-nucbp];
    }
    if (idown != -1) {
      int dpos = pos-nucpos[idown];
      if (dpos < nucbp) return; // Nucleosomes cannot overlap
      else if (dpos < llink+nucbp) dErep += erep[dpos-nucbp];
    }
    if (p < min(1.0, nbp/static_cast<double>((nnuc+1.0)*nucbp)*
		exp((mu-emeth[pos]-dErep)/temp))) {
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
      if (dpos < llink+nucbp) dErep -= erep[dpos-nucbp];
    }
    if (idown != -1) {
      int dpos = pos-nucpos[idown];
      if (dpos < llink+nucbp) dErep -= erep[dpos-nucbp];
    }
    if (p < min(1.0, (nnuc*nucbp)/static_cast<double>(nbp)*
		exp((-mu+emeth[pos]-dErep)/temp))) {
      nucpos.erase(nucpos.begin()+inuc);
    }
  }
}

void NucPosModel::run(lint nsweep, double startTemp, double endTemp,
		      Cooling coolOption) {
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
    for (int i = 0; i < maxNumOfNuc; i++) {
      update();
    }
    output(n+1);
  }
  for (auto& t : trackers) t->finalize(nsweep, *this); 
}

void NucPosModel::output(lint time) {
  for (auto& t : trackers) {
    t->track(time, *this);
  }
}

const vector<int>& NucPosModel::getNucPos() const {
  return nucpos;
}

double NucPosModel::getEnergy() const {
  double totalEmeth = 0.0;
  double totalErep = 0.0;  
  for (size_t i = 0; i < nucpos.size(); i++) {
    int pos = nucpos[i];
    totalEmeth += emeth[pos];
    if (i > 0) {
      int dpos = nucpos[i]-nucpos[i-1];
      if (dpos < nucbp+llink) totalErep += erep[dpos-nucbp];
    }
  }
  return totalEmeth + totalErep - mu*nucpos.size();
}

double NucPosModel::getTemp() const {
  return temp;
}

const vector<double>& NucPosModel::getMethEnergy() const {
  return emeth;
}

const NucPosModel::Params& NucPosModel::getParams() const {
  return params;
}

void NucPosModel::addTracker(std::shared_ptr<Tracker> tracker) {
  trackers.push_back(tracker);
}

double min(double a, double b) {
  return (a < b ? a : b);
}
