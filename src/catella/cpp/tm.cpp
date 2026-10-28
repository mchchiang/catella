// tm.cpp

#include <algorithm>
#include <cmath>
#include <limits>
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
			     "results");
  }
}

const vector<double>& NucPosTM::getSeqEnergy() const {
  return model.eseq();
}

void NucPosTM::solve(double t) {
  if (!(t > 0)) {
    throw std::runtime_error("temp must be positive");
  }
  temp = t;
  computeForwardFull();
  computeBackwardAndMarginals();
  computeOccupFromStartProb();
  meanN = std::accumulate(startProb.begin(), startProb.end(), 0.0);
  solved = true;
}

double NucPosTM::logPlaceWeight(int x, int s, double t, double mu) const {
  return -(model.eseq()[x] + model.erep(s) - mu)/t;
}

namespace {

const double NINF = -std::numeric_limits<double>::infinity();

double logAdd(double a, double b) {
  if (a < b) std::swap(a, b);
  if (a == NINF) return NINF;
  return a + std::log1p(std::exp(b - a));
}

double logSumExp(const vector<double>& a) {
  double m = *std::max_element(a.begin(), a.end());
  if (m == NINF) return NINF;
  double sum = 0.0;
  for (double e : a) sum += std::exp(e - m);
  return m + std::log(sum);
}

void requireFinite(double logValue, const char* pass) {
  if (std::isnan(logValue) || (std::isinf(logValue) && logValue > 0)) {
    throw std::runtime_error(string("NucPosTM: ") + pass +
			     " pass produced a non-finite weight");
  }
}

}

void NucPosTM::forwardStep(const vector<double>& cur, vector<double>& nxt,
			   int x, double t, double mu) const {
  int W = model.nucbp() + model.llink();
  int nucbp = model.nucbp();

  vector<double> logPlace(W - nucbp + 1);
  for (int k = nucbp-1; k < W; k++) {
    logPlace[k-nucbp+1] = cur[k] + logPlaceWeight(x, k+1, t, mu);
  }
  nxt[0] = logSumExp(logPlace);
  for (int k = 1; k < W-1; k++) nxt[k] = cur[k-1];
  nxt[W-1] = logAdd(cur[W-2], cur[W-1]);
  requireFinite(nxt[0], "forward");
}

void NucPosTM::computeForwardFull() {
  int W = model.nucbp() + model.llink();
  int npos = model.npos();
  double mu = model.mu();

  logV.assign(npos+1, vector<double>(W, NINF));
  logV[0][W-1] = 0.0; // s = W (no nucleosome to the left)

  for (int x = 0; x < npos; x++) {
    forwardStep(logV[x], logV[x+1], x, temp, mu);
  }
  logZ = logSumExp(logV[npos]);
}

void NucPosTM::computeBackwardAndMarginals() {
  int W = model.nucbp() + model.llink();
  int npos = model.npos();
  int nucbp = model.nucbp();
  double mu = model.mu();

  logU.assign(npos+1, vector<double>(W, 0.0));
  startProb.assign(npos, 0.0);
  vector<double> logW(W, NINF);
  vector<double> logStart(W - nucbp + 1);

  for (int x = npos-1; x >= 0; x--) {
    const vector<double>& nxtU = logU[x+1];
    for (int k = nucbp-1; k < W; k++) {
      logW[k] = logPlaceWeight(x, k+1, temp, mu);
      logStart[k-nucbp+1] = logV[x][k] + logW[k];
    }
    startProb[x] = std::exp(logSumExp(logStart) + nxtU[0] - logZ);

    vector<double>& cur = logU[x];
    for (int k = 0; k < W; k++) {
      cur[k] = nxtU[std::min(k+1, W-1)];
      if (k+1 >= nucbp) cur[k] = logAdd(cur[k], logW[k] + nxtU[0]);
    }
    requireFinite(cur[W-1], "backward");
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

  vector<double> cur(W, NINF);
  cur[W-1] = 0.0;
  vector<double> nxt(W);

  for (int x = 0; x < npos; x++) {
    forwardStep(cur, nxt, x, t, muOverride);
    std::swap(cur, nxt);
  }
  return logSumExp(cur);
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

  vector<int> positions;
  int s = W;
  for (int x = 0; x < npos; x++) {
    bool placed = false;
    if (s >= nucbp) {
      double logPlace = logPlaceWeight(x, s, temp, mu) + logU[x+1][0];
      double logSkip = logU[x+1][std::min(s, W-1)];
      double pPlace = 1.0/(1.0 + std::exp(logSkip - logPlace));
      double r = rand(mt);
      if (r < pPlace) placed = true;
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
