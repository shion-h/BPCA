//
// BPCA.cpp
//
// Copyright (c) 2019 Shion Hosoda
//
// This software is released under the MIT License.
// http://opensource.org/licenses/mit-license.php
//

//include{{{
#include <stdlib.h>
#include <boost/program_options.hpp>
#include "include/CsvFileParser.hpp"
#include "include/VariationalInferencerOfBPCA.hpp"
//}}}

using namespace std;

int main(int argc, char *argv[]){

    //Options{{{
    boost::program_options::options_description opt("Options");
    opt.add_options()
    ("help,h", "show help")
    ("output,o", boost::program_options::value<string>()->default_value("./"), "Directory name for output")
    ("threshold-convergence,c", boost::program_options::value<double>()->default_value(0.0), "Threshold of convergence")
    ("threshold-shrinkage,s", boost::program_options::value<double>()->default_value(0.0001), "Threshold of shrinkage of alpha")
    ("iteration-number,n", boost::program_options::value<int>()->default_value(0), "The number of iteration")
    ("dimension,d", boost::program_options::value<int>()->default_value(10), "The initial number of dimension of Y")
    ;

    boost::program_options::positional_options_description pd;
    pd.add("xfile", 1);

    boost::program_options::options_description hidden("hidden");
    hidden.add_options()
        ("xfile", boost::program_options::value<string>(), "hidden")
        ;
    boost::program_options::options_description cmdline_options;
    cmdline_options.add(opt).add(hidden);

    boost::program_options::variables_map vm;
    try{
        boost::program_options::store(boost::program_options::command_line_parser(argc, argv).options(cmdline_options).positional(pd).run(), vm);
    }catch(const boost::program_options::error_with_option_name& e){
        cout<<e.what()<<endl;
    }
    boost::program_options::notify(vm);
    string xFilename;
    double thresholdOfConvergence, thresholdOfShrinkage;
    int iterNum;
    int K;
    string outputDirectory;
    if (vm.count("help") || !vm.count("xfile")){
        cout<<"Usage:\n BPCA [data file] [-options] "<<endl;
        cout<<endl;
        cout<<opt<<endl;
        exit(1);
    }else{
        xFilename = vm["xfile"].as<std::string>();
        if(vm.count("output"))outputDirectory = vm["output"].as<std::string>();
        if(outputDirectory[outputDirectory.size()-1] != '/')outputDirectory.push_back('/');
        if(vm.count("threshold-convergence"))thresholdOfConvergence = vm["threshold-convergence"].as<double>();
        if(vm.count("threshold-shrinkage"))thresholdOfShrinkage = vm["threshold-shrinkage"].as<double>();
        if(vm.count("iteration-number"))iterNum = vm["iteration-number"].as<int>();
        if(vm.count("dimension"))K = vm["dimension"].as<int>();
    }

    CsvFileParser<double> xFile(xFilename);
    //For debug
    // CsvFileParser<double> yFile("../data/y.csv");
    // CsvFileParser<double> wtFile("../data/wt.csv");
    // CsvFileParser<double> muFile("../data/mu.csv");
    // CsvFileParser<double> sigma2File("../data/sigma2.csv");
    // CsvFileParser<double> alphaFile("../data/alpha.csv");

    //}}}

//estimation{{{
    VariationalInferencerOfBPCA *estimator;
    estimator = new VariationalInferencerOfBPCA(xFile, K, thresholdOfShrinkage, thresholdOfConvergence, iterNum);
    //For debug
    // estimator = new VariationalInferencerOfBPCA(xFile, K, thresholdOfShrinkage, thresholdOfConvergence, iterNum, yFile, wtFile, muFile, sigma2File, alphaFile);
    estimator->runIteraions();
    estimator->writeParameters(outputDirectory);
    estimator->writeELBO(outputDirectory);
    delete estimator;
//}}}
    return 0;
}
