// tm.cpp

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include "tm.hpp"

using std::string;
using std::vector;

NucPosTM::NucPosTM(int nucbp, int nbp, int llink, double elink, double mu,
		   ulint seed_) :
  model(nucbp, nbp, llink, elink, mu), seed(seed_), temp(1.0),
  solved(false), logZ(0.0), meanN(0.0), mt(seed_),
  rand(0.0, 1.0) {}

NucPosTM::NucPosTM(const NucPosModel::Params& params, ulint seed_) :
  model(params), seed(seed_), temp(1.0), solved(false), logZ(0.0),
  meanN(0.0), mt(seed_), rand(0.0, 1.0) {}

NucPosTM::~NucPosTM() {}

void NucPosTM::reset() {
  mt = std::mt19937(seed);
}

void NucPosTM::setEnergy(string dataFile, double emax) {
  model.setEnergy(dataFile, emax);
  solved = false;
}

void NucPosTM::setEnergy(const vector<double>& eseq, double emax) {
  model.setEnergy(eseq, emax);
  solved = false;
}

void NucPosTM::requireSolved() const {
  if (!solved) {
    throw std::runtime_error("NucPosTM: call solve() before querying "
			     "results.");
  }
}

const vector<double>& NucPosTM::getSeqEnergy() const {
  return model.eseq();
}

void NucPosTM::solve(double t) {
  if (!(t > 0)) {
    throw std::runtime_error("temp must be positive.");
  }
  temp = t;
  computeForwardFull();
  computeBackwardAndMarginals();
  computeOccupFromStartProb();
  meanN = std::accumulate(startProb.begin(), startProb.end(), 0.0);
  solved = true;
}

void NucPosTM::computeForwardFull() {
  int W = model.nucbp() + model.llink();
  int npos = model.npos();
  int nucbp = model.nucbp();
  double mu = model.mu();
  const vector<double>& eseq = model.eseq();

  v.assign(npos+1, vector<double>(W, 0.0));
  logScaleV.assign(npos+1, 0.0);
  v[0][W-1] = 1.0; // s = W (no nucleosome to the left)

  for (int x = 0; x < npos; x++) {
    const vector<double>& cur = v[x];
    vector<double>& nxt = v[x+1];
    for (int k = 0; k < W; k++) {
      double val = cur[k];
      if (val == 0.0) continue;
      int s = k+1;
      int kSkip = std::min(k+1, W-1);
      nxt[kSkip] += val;
      if (s >= nucbp) {
	double w = std::exp(-(eseq[x] + model.erep(s) - mu)/temp);
	nxt[0] += val * w;
      }
    }
    double scale = std::accumulate(nxt.begin(), nxt.end(), 0.0);
    if (!(scale > 0.0)) {
      throw std::runtime_error("NucPosTM: forward pass underflowed to "
			       "zero.");
    }
    for (double& e : nxt) e /= scale;
    logScaleV[x+1] = logScaleV[x] + std::log(scale);
  }
  logZ = logScaleV[npos] +
    std::log(std::accumulate(v[npos].begin(), v[npos].end(), 0.0));
}

void NucPosTM::computeBackwardAndMarginals() {
  int W = model.nucbp() + model.llink();
  int npos = model.npos();
  int nucbp = model.nucbp();
  double mu = model.mu();
  const vector<double>& eseq = model.eseq();

  u.assign(npos+1, vector<double>(W));
  logScaleU.assign(npos+1, 0.0);
  std::fill(u[npos].begin(), u[npos].end(), 1.0/W);
  logScaleU[npos] = std::log(static_cast<double>(W));

  startProb.assign(npos, 0.0);

  for (int x = npos-1; x >= 0; x--) {
    // Combine v[x] with u[x+1] to get the marginal start probability.
    double sumTerm = 0.0;
    for (int k = nucbp-1; k < W; k++) {
      int s = k+1;
      double w = std::exp(-(eseq[x] + model.erep(s) - mu)/temp);
      sumTerm += v[x][k] * w;
    }
    double logCombined = logScaleU[x+1] + logScaleV[x] - logZ;
    startProb[x] = std::exp(logCombined) * u[x+1][0] * sumTerm;

    // Step u backward from x+1 to x.
    vector<double>& nxt = u[x];
    for (int k = 0; k < W; k++) {
      int s = k+1;
      int kSkip = std::min(k+1, W-1);
      double val = u[x+1][kSkip];
      if (s >= nucbp) {
	double w = std::exp(-(eseq[x] + model.erep(s) - mu)/temp);
	val += w * u[x+1][0];
      }
      nxt[k] = val;
    }
    double scale = std::accumulate(nxt.begin(), nxt.end(), 0.0);
    if (!(scale > 0.0)) {
      throw std::runtime_error("NucPosTM: backward pass underflowed to "
			       "zero.");
    }
    for (double& e : nxt) e /= scale;
    logScaleU[x] = logScaleU[x+1] + std::log(scale);
  }
}

void NucPosTM::computeOccupFromStartProb() {
  int nbp = model.nbp();
  int nucbp = model.nucbp();
  int npos = model.npos();

  vector<double> prefix(npos+1, 0.0);
  for (int x = 0; x < npos; x++) prefix[x+1] = prefix[x] + startProb[x];

  occup.assign(nbp, 0.0);
  for (int y = 0; y < nbp; y++) {
    int xLo = std::max(0, y - nucbp + 1);
    int xHi = std::min(y, npos-1);
    if (xHi < xLo) continue;
    occup[y] = prefix[xHi+1] - prefix[xLo];
  }
}

double NucPosTM::forwardLogZOnly(double t, double muOverride) const {
  int W = model.nucbp() + model.llink();
  int npos = model.npos();
  int nucbp = model.nucbp();
  const vector<double>& eseq = model.eseq();

  vector<double> cur(W, 0.0);
  cur[W-1] = 1.0;
  vector<double> nxt(W);
  double logScale = 0.0;

  for (int x = 0; x < npos; x++) {
    std::fill(nxt.begin(), nxt.end(), 0.0);
    for (int k = 0; k < W; k++) {
      double val = cur[k];
      if (val == 0.0) continue;
      int s = k+1;
      int kSkip = std::min(k+1, W-1);
      nxt[kSkip] += val;
      if (s >= nucbp) {
	double w = std::exp(-(eseq[x] + model.erep(s) - muOverride)/t);
	nxt[0] += val * w;
      }
    }
    double scale = std::accumulate(nxt.begin(), nxt.end(), 0.0);
    if (!(scale > 0.0)) {
      throw std::runtime_error("NucPosTM: forward pass underflowed to "
			       "zero.");
    }
    for (double& e : nxt) e /= scale;
    logScale += std::log(scale);
    std::swap(cur, nxt);
  }
  return logScale + std::log(std::accumulate(cur.begin(), cur.end(), 0.0));
}

double NucPosTM::getLogZ() const {
  requireSolved();
  return logZ;
}

const vector<double>& NucPosTM::getStartProb() const {
  requireSolved();
  return startProb;
}

const vector<double>& NucPosTM::getOccup() const {
  requireSolved();
  return occup;
}

double NucPosTM::getMeanN() const {
  requireSolved();
  return meanN;
}

double NucPosTM::getVarN(double dmu) const {
  // Self-contained: only needs model.mu()/model.eseq() and temp (which
  // defaults to 1.0), not any cached solve() state.
  double mu = model.mu();
  double lp = forwardLogZOnly(temp, mu + dmu);
  double lm = forwardLogZOnly(temp, mu - dmu);
  double l0 = forwardLogZOnly(temp, mu);
  double d2 = (lp - 2.0*l0 + lm) / (dmu*dmu);
  return temp*temp*d2;
}

vector<double> NucPosTM::getVarOccup() const {
  requireSolved();
  vector<double> var(occup.size());
  for (size_t y = 0; y < occup.size(); y++) {
    var[y] = occup[y] * (1.0 - occup[y]);
  }
  return var;
}

double NucPosTM::configEnergy(const vector<int>& positions) const {
  double totalEseq = 0.0;
  double totalErep = 0.0;
  const vector<double>& eseq = model.eseq();
  for (size_t i = 0; i < positions.size(); i++) {
    totalEseq += eseq[positions[i]];
    if (i > 0) {
      totalErep += model.erep(positions[i] - positions[i-1]);
    }
  }
  return totalEseq + totalErep - model.mu()*positions.size();
}

std::pair<vector<int>, double> NucPosTM::sampleConfig() {
  requireSolved();
  int W = model.nucbp() + model.llink();
  int npos = model.npos();
  int nucbp = model.nucbp();
  double mu = model.mu();
  const vector<double>& eseq = model.eseq();

  vector<int> positions;
  int s = W;
  for (int x = 0; x < npos; x++) {
    bool placed = false;
    if (s >= nucbp) {
      double w = std::exp(-(eseq[x] + model.erep(s) - mu)/temp);
      double piPlace = w * u[x+1][0];
      int kSkip = std::min(s, W-1);
      double piSkip = u[x+1][kSkip];
      double r = rand(mt);
      if (r < piPlace/(piPlace+piSkip)) placed = true;
    }
    if (placed) {
      positions.push_back(x);
      s = 1;
    } else {
      s = std::min(s+1, W);
    }
  }
  double energy = configEnergy(positions);
  return {positions, energy};
}
