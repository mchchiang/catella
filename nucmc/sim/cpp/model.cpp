// model.cpp

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <set>
#include <cmath>
#include <random>
#include <algorithm>
#include <memory>
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
			 long _seed) :
  nucbp(_nucbp), nbp(_nbp), llink(_llink), mu(_mu), seed(_seed) {

  // Precompute repulsion strengths - WCA repulsion
  erep = vector<double>(llink, 0.0);
  double sigma = llink/(pow(2.0,(1.0/6.0)));
  for (int i = 0; i < llink; i++) {
    double sr6 = pow(sigma/(i+1.0),6.0);
    erep[i] = 4*(sr6*sr6-sr6+0.25);
  }
  maxNumOfNuc = nbp/nucbp;
  npos = nbp-nucbp;

  reset();
}

NucPosModel::~NucPosModel() {}

void NucPosModel::reset() {
  // No nucleosomes on the fibre initially
  nucpos = vector<int>();

  // Set up random generator
  mt = std::mt19937(seed);  
  randMode = std::uniform_int_distribution<int>(0,2);
  randPos = std::uniform_int_distribution<int>(0,npos-1);
  rand = std::uniform_real_distribution<double>(0.0,1.0);

  // Reset the temperature
  temp = 1.0;

  // Reset energy landscape
  emeth = vector<double>(nbp, 0.0);  
}

void NucPosModel::initByMethData(string dataFile) {
  emeth = vector<double>(nbp, 0.0);
  ifstream reader;
  reader.open(dataFile);
  if (!reader) {
    cout << "ERROR: cannot open the file " << dataFile << endl;
    exit(1);
  }
  string line;
  stringstream ss;
  int pos;
  double score;
  while (getline(reader, line)) {
    if (line[0] == '#') continue; // Skip comments
    ss.clear();
    ss.str(line);
    ss >> pos >> score;
    if (pos >= 0 && pos < nbp) {
      emeth[pos] = score;
    }
  }
  reader.close();
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

    // Skip if shifting the nucleosome causes it to move off the fibre
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

void NucPosModel::run(int nsweeps, double startTemp, double endTemp,
		      int nincs) {
  double tempInc;
  int nsweepsPerTemp;
  if (nincs <= 0) {
    tempInc = 0.0;
    startTemp = endTemp;
    nsweepsPerTemp = nsweeps;
  } else {
    tempInc = (endTemp-startTemp)/(nincs+1.0);
    // Allow time for the start and end temp    
    nsweepsPerTemp = static_cast<int>(ceil(nsweeps/(nincs+1.0)));
  }
  temp = startTemp;
  for (int n = 0; n < nsweeps; n++) {
    if (n % 100 == 0) {
      cout << "Working on t = " << n << " T = " << temp << endl;
    }
    output(n);
    for (int i = 0; i < maxNumOfNuc; i++) {
      update();
    }
    if ((n+1) % nsweepsPerTemp == 0) {
      temp += tempInc;
    }    
  }
  if (nsweeps % 100 == 0) {
    cout << "Working on t = " << nsweeps << " T = " << temp << endl;
  }  
  output(nsweeps);
}

void NucPosModel::output(int time) {
  for (auto& t : trackers) {
    t->track(time, *this);
  }
}

const vector<int>& NucPosModel::getNucPos() const {
  return nucpos;
}

int NucPosModel::getNucbp() const {
  return nucbp;
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

void NucPosModel::addTracker(std::shared_ptr<Tracker> tracker) {
  trackers.push_back(tracker);
}

double min(double a, double b) {
  return (a < b ? a : b);
}
