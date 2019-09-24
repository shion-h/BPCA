//
// VariationalInferencerOfBPCA.hpp
//
// Copyright (c) 2019 Shion Hosoda
//
// This software is released under the MIT License.
// http://opensource.org/licenses/mit-license.php
//

#include<stdlib.h>
#include<math.h>
#include<cmath>
#include<iostream>
#include<vector>
#include<numeric>
#include<memory>
#include<random>
#include<iomanip>
#include<fstream>
#include<limits>
#include<algorithm>
#include<map>
#include<Eigen/Dense>
#include<Eigen/Core>
#include<Eigen/LU>
#include"CsvFileParser.hpp"
#include"utils.hpp"


void removeRow(Eigen::MatrixXd& matrix, unsigned int rowToRemove);
void removeColumn(Eigen::MatrixXd& matrix, unsigned int colToRemove);

class VariationalInferencerOfBPCA{
protected:
    const Eigen::MatrixXd &_x;
    const int _N, _D;
    int _K;
    Eigen::MatrixXd _exY, _exYYT, _covY, _exWT, _exWTW, _covWT;
    Eigen::VectorXd _alpha, _mu;
    std::vector<Eigen::VectorXd> _alphaTrace;
    double _sigma2;
    std::vector<double> _ELBO;
    double _thresholdOfShrinkage, _thresholdOfConvergence;
    int _iterNum;
public:
    VariationalInferencerOfBPCA(const CsvFileParser<double> &xFile, int K, double thresholdOfShrinkage, double thresholdOfConvergence, int iterNum);
    VariationalInferencerOfBPCA(const CsvFileParser<double> &xFile, int K, double thresholdOfShrinkage, double thresholdOfConvergence, int iterNum, const CsvFileParser<double> &yFile, const CsvFileParser<double> &wtFile, const CsvFileParser<double> &muFile, const CsvFileParser<double> &sigma2File, const CsvFileParser<double> &alphaFile);
    virtual ~VariationalInferencerOfBPCA();
    virtual void initializeParameters(bool hasInitialY=false, bool hasInitialWT=false, bool hasInitialMu=false, bool hasInitialSigma2=false, bool hasInitialAlpha=false);
    virtual void updateY();
    virtual void updateW();
    virtual void updateMu();
    virtual void updateSigma2();
    virtual void updateAlpha();
    virtual void updateHyperParams();
    virtual bool hasNonRelevantComponent(int k, double threshold)const;
    virtual void checkARD();
    virtual void shrinkComponent(int k);
    virtual void calculateELBO();
    virtual void writeParameters(std::string outputDirectory)const;
    virtual void writeELBO(std::string outputDirectory)const;
    virtual bool isConvergence()const;
    virtual void runIteraions();
};
