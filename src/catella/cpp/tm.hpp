// tm.hpp
//
// Exact transfer-matrix solver for the nucleosome positioning model,
// as an alternative to the Metropolis sampler in NucPosMCMC.

#ifndef TM_HPP
#define TM_HPP

#include <random>
#include <string>
#include <utility>
#include <vector>
#include "dtype.hpp"
#include "model.hpp"

class NucPosTM {

public:
  NucPosTM(int nucbp, int nbp, int llink, double elink, double mu,
	   ulint seed);
  explicit NucPosTM(const NucPosModel::Params& params, ulint seed);
  ~NucPosTM();

  void setEnergy(std::string dataFile, double emax);
  void setEnergy(const std::vector<double>& eseq, double emax);

  // Runs the forward-backward solve. Call before any query method
  // below, and again after any setEnergy call.
  void solve(double temp = 1.0);

  double getLogZ() const;
  const std::vector<double>& getStartProb() const;
  const std::vector<double>& getOccup() const;
  double getMeanN() const;

  // Central finite difference of <N> = temp * d(logZ)/d(mu).
  double getVarN(double dmu = 1e-4) const;

  // Closed form: occupancy at a single bp is Bernoulli (nucleosomes
  // cannot overlap), so Var = occup*(1-occup) exactly.
  std::vector<double> getVarOccup() const;

  // One independent exact sample via stochastic traceback: sampled
  // start positions plus that configuration's exact energy.
  std::pair<std::vector<int>, double> sampleConfig();

  const std::vector<double>& getSeqEnergy() const;

  // Re-seeds the sampling RNG from the original seed.
  void reset();

  // Replaces the seed and re-seeds the sampling RNG from it.
  void setSeed(ulint seed);

private:
  NucPosModel model;
  ulint seed;
  double temp;
  bool solved;

  // logV[x][k] / logU[x][k]: log forward/backward vectors, k=0..W-1
  // representing renewal state s=k+1 (s=0 never occurs).
  std::vector<std::vector<double> > logV;
  std::vector<std::vector<double> > logU;

  double logZ;
  double meanN;
  std::vector<double> startProb;
  std::vector<double> occup;

  std::mt19937 mt;
  std::uniform_real_distribution<double> rand;

  void computeForwardFull();
  void computeBackwardAndMarginals();
  void computeOccupFromStartProb();
  double forwardLogZOnly(double t, double muOverride) const;
  double logPlaceWeight(int x, int s, double t, double mu) const;
  void forwardStep(const std::vector<double>& cur,
		   std::vector<double>& nxt, int x, double t,
		   double mu) const;
  double configEnergy(const std::vector<int>& positions) const;
  void requireSolved() const;
};

#endif
