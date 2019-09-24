//
// VariationalInferencerOfBPCA.cpp
//
// Copyright (c) 2019 Shion Hosoda
//
// This software is released under the MIT License.
// http://opensource.org/licenses/mit-license.php
//
#include"include/VariationalInferencerOfBPCA.hpp"

using namespace std;

using namespace Eigen;

random_device rd;
mt19937 gen(rd());

void removeElement(VectorXd& vector, unsigned int indexToRemove){//{{{
    vector.segment(indexToRemove, vector.size() - 1 - indexToRemove) = vector.tail(vector.size() - 1 - indexToRemove);
    vector.conservativeResize(vector.size()-1, 1);
}//}}}

void removeRow(Eigen::MatrixXd& matrix, unsigned int rowToRemove){//{{{
    unsigned int numRows = matrix.rows()-1;
    unsigned int numCols = matrix.cols();

    if( rowToRemove < numRows )
        matrix.block(rowToRemove,0,numRows-rowToRemove,numCols) = matrix.bottomRows(numRows-rowToRemove);

    matrix.conservativeResize(numRows,numCols);
}//}}}

void removeColumn(Eigen::MatrixXd& matrix, unsigned int colToRemove){//{{{
    unsigned int numRows = matrix.rows();
    unsigned int numCols = matrix.cols()-1;

    if( colToRemove < numCols )
        matrix.block(0,colToRemove,numRows,numCols-colToRemove) = matrix.rightCols(numCols-colToRemove);

    matrix.conservativeResize(numRows,numCols);
}//}}}

VariationalInferencerOfBPCA::VariationalInferencerOfBPCA(const CsvFileParser<double> &xFile, int K, double thresholdOfShrinkage, double thresholdOfConvergence, int iterNum)//{{{
    :
     _x(xFile.getEigenMatrix()),
     _N(_x.cols()),
     _D(_x.rows()),
     _K(K),
     _exY(_K, _N),
     _exYYT(_K, _K),
     _covY(MatrixXd::Identity(_K, _K)),
     _exWT(_K, _D),
     _exWTW(_K, _K),
     _covWT(MatrixXd::Identity(_K, _K)),
     _alpha(_K),
     _mu(_D),
     _thresholdOfShrinkage(thresholdOfShrinkage),
     _thresholdOfConvergence(thresholdOfConvergence),
     _iterNum(iterNum)
{
    this->initializeParameters();
}//}}}

VariationalInferencerOfBPCA::VariationalInferencerOfBPCA(const CsvFileParser<double> &xFile, int K, double thresholdOfShrinkage, double thresholdOfConvergence, int iterNum, const CsvFileParser<double> &yFile, const CsvFileParser<double> &wtFile, const CsvFileParser<double> &muFile, const CsvFileParser<double> &sigma2File, const CsvFileParser<double> &alphaFile)//{{{
    :
     _x(xFile.getEigenMatrix()),
     _N(_x.cols()),
     _D(_x.rows()),
     _K(K),
     _exY(yFile.getEigenMatrix()),
     _exYYT(_K, _K),
     _covY(MatrixXd::Identity(_K, _K)),
     _exWT(wtFile.getEigenMatrix()),
     _exWTW(_K, _K),
     _covWT(MatrixXd::Identity(_K, _K)),
     _alpha(alphaFile.getEigenMatrix()),
     _mu(muFile.getEigenMatrix()),
     _sigma2(sigma2File.getEigenMatrix()(0, 0)),
     _thresholdOfShrinkage(thresholdOfShrinkage),
     _thresholdOfConvergence(thresholdOfConvergence),
     _iterNum(iterNum)
{
    this->initializeParameters(false, false, true, true, true);
}//}}}

VariationalInferencerOfBPCA::~VariationalInferencerOfBPCA(){//{{{

}//}}}

void VariationalInferencerOfBPCA::initializeParameters(bool hasInitialY, bool hasInitialWT, bool hasInitialMu, bool hasInitialSigma2, bool hasInitialAlpha){//{{{
    normal_distribution<float> dis(0.0, 1.0);
    uniform_real_distribution<float> disPos(0, 10);

    if(!hasInitialY){
        _exY = MatrixXd::NullaryExpr(_K, _N, [&](){return dis(gen);});
    }
    _exYYT = _exY * _exY.transpose() + _N * _covY;

    if(!hasInitialWT){
        _exWT = MatrixXd::NullaryExpr(_K, _D, [&](){return dis(gen);});
    }
    _exWTW = _exWT * _exWT.transpose() + _D * _covWT;

    // cov won't be initialized.

    if(!hasInitialMu){
        // _mu = VectorXd::NullaryExpr(_D, [&](){return disPos(gen);});
        _mu = _x.rowwise().sum() / _N;
    }
    if(!hasInitialSigma2){
        _sigma2 = disPos(gen);
    }
    if(!hasInitialAlpha){
        _alpha = VectorXd::NullaryExpr(_K, [&](){return disPos(gen);});
    }
}//}}}

void VariationalInferencerOfBPCA::updateY(){//{{{
    // covY is shared by y_is.
    MatrixXd precY = MatrixXd::Identity(_K, _K) + _exWTW / _sigma2;
    _covY = precY.inverse();
    MatrixXd T = MatrixXd::Zero(_K, _N);
    T = _exWT * (_x.colwise() - _mu) / _sigma2;
    _exY = _covY * T;
    _exYYT = _exY * _exY.transpose() + _N * _covY;
}//}}}

void VariationalInferencerOfBPCA::updateW(){//{{{
    MatrixXd alphaMatrix(_alpha.asDiagonal());
    MatrixXd precW = alphaMatrix.inverse() + _exYYT / _sigma2;
    _covWT = precW.inverse();
    MatrixXd T = MatrixXd::Zero(_K, _D);
    T = _exY * (_x.colwise() - _mu).transpose() / _sigma2;
    _exWT = _covWT * T;
    _exWTW = _exWT * _exWT.transpose() + _D * _covWT;
}//}}}

void VariationalInferencerOfBPCA::updateMu(){//{{{
    _mu = (_x.rowwise().sum() - _exWT.transpose() * _exY.rowwise().sum()) / _N;
}//}}}

void VariationalInferencerOfBPCA::updateSigma2(){//{{{
    _sigma2 = 0;
    MatrixXd mat1 = (_x.colwise() - _mu).transpose();
    MatrixXd mat2 = (_x - 2*_exWT.transpose()*_exY).colwise() - _mu;
    for(int i=0; i<_N; i++){
        _sigma2 += mat1.row(i) * mat2.col(i);
        _sigma2 += _exY.col(i).transpose() * _exWTW * _exY.col(i);
    }
    _sigma2 += _N * (_covY.array() * _exWTW.array()).sum();
    _sigma2 /= _D * _N;
}//}}}

void VariationalInferencerOfBPCA::updateAlpha(){//{{{
    _alpha = _exWTW.diagonal() / _D;
    _alphaTrace.push_back(_alpha);
}//}}}

void VariationalInferencerOfBPCA::updateHyperParams(){//{{{
    this->updateMu();
    this->updateSigma2();
    this->updateAlpha();
}//}}}

bool VariationalInferencerOfBPCA::hasNonRelevantComponent(int k, double threshold)const{//{{{
    if(_alpha(k) < threshold){
        return true;
    }else{
        return false;
    }
}//}}}

void VariationalInferencerOfBPCA::shrinkComponent(int k){//{{{
    removeRow(_exY, k);
    removeRow(_exYYT, k);
    removeRow(_covY, k);
    removeRow(_exWT, k);
    removeRow(_exWTW, k);
    removeRow(_covWT, k);
    removeColumn(_exYYT, k);
    removeColumn(_covY, k);
    removeColumn(_exWTW, k);
    removeColumn(_covWT, k);
    removeElement(_alpha, k);
    _K = _K - 1;
}//}}}

void VariationalInferencerOfBPCA::checkARD(){//{{{
    // considering shifts of indices
    // double threshold = _alpha.maxCoeff() / _thresholdOfShrinkage;
    // double threshold = _thresholdOfShrinkage / _K;
    double threshold = _thresholdOfShrinkage;
    for(int k=_K-1; k>=0; k--){
        if(this->hasNonRelevantComponent(k, threshold)){
            this->shrinkComponent(k);
            // _alpha(k) = 0.01;
            if(_K == 0)break;
        }
    }
}//}}}

void VariationalInferencerOfBPCA::calculateELBO(){//{{{
    double thisELBO = 0.0;
    double sumLnAlpha = _alpha.array().log().sum();
    thisELBO += - _D*sumLnAlpha/2 - _D*_N*log(2*M_PI*_sigma2)/2;
    thisELBO += (_N*_K + _K*_D - _D*_N)/2 + _N*log(_covY.determinant())/2 + _D*log(_covWT.determinant())/2;
    double sumYTY = _exY.squaredNorm() + _N * _covY.trace();
    double sumAlphaWTW = 0.0;
    for(int k=0; k<_K; k++){
        sumAlphaWTW += (_exWT.row(k) * _exWT.row(k).transpose())(0,0) / _alpha(k);
        sumAlphaWTW += _D * _covWT(k, k) / _alpha(k);
    }
    thisELBO += - sumYTY / 2 - sumAlphaWTW / 2;
    _ELBO.push_back(thisELBO);
}//}}}

void VariationalInferencerOfBPCA::writeParameters(string outputDirectory)const{//{{{
    string covYFilename, exYFilename, covWTFilename, exWTFilename, alphaFilename;
    covYFilename = outputDirectory + "covY.csv";
    exYFilename = outputDirectory + "exY.csv";
    covWTFilename = outputDirectory + "covWT.csv";
    exWTFilename = outputDirectory + "exWT.csv";
    alphaFilename = outputDirectory + "alpha.csv";
    if(_K != 0){
        outputEigenMatrix(_covY, covYFilename);
        outputEigenMatrix(_exY, exYFilename);
        outputEigenMatrix(_covWT, covWTFilename);
        outputEigenMatrix(_exWT, exWTFilename);
        outputVectorEigenVector(_alphaTrace, alphaFilename);
    }
}//}}}

void VariationalInferencerOfBPCA::writeELBO(string outputDirectory)const{//{{{
    string ELBOFilename;
    ELBOFilename = outputDirectory + "ELBO.csv";
    outputVector(_ELBO, ELBOFilename);
}//}}}

bool VariationalInferencerOfBPCA::isConvergence()const{//{{{
    double thisELBO = _ELBO[_ELBO.size()-1];
    double prevELBO = _ELBO[_ELBO.size()-2];
    double rate = (thisELBO - prevELBO) / abs(prevELBO);
    if(rate < _thresholdOfConvergence){
        return true;
    }else{
        return false;
    }
}//}}}

void VariationalInferencerOfBPCA::runIteraions(){//{{{
    int c = 0;
    while(1){
        this->updateY();
        this->updateW();
        this->updateHyperParams();
        this->calculateELBO();
        cout<< '\r'<< "Loop: #"<<c<< string(20, ' ');
        this->checkARD();
        if(_K == 0)break;
        c++;
        // if(thresholdOfConvergence!=0 && this->isConvergence())break;
        if(c > _iterNum){
            cout<<endl;
            break;
        }
    }
}//}}}
