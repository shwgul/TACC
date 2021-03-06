#include <iostream>
#include <cstring>
#include <string>
#include <rose.h>
#include <stdio.h>
#include <fstream>
#include <vector>
#include <rewrite.h>
#include "LinearTranslator.h"
#include "Data.h"

using namespace SageInterface;
using namespace SageBuilder;
using namespace AstFromString;
using namespace std;

void LinearTranslator::readFile() {
  ifstream file ("../GenData/cudaInData.txt");
  string line;

  if (file.is_open()) {
    for(int i = 0; i < transData.device_variable.size(); i++) {
	getline (file, line);
    	transData.cuda_io.push_back(line);
	cout << transData.cuda_io.at(i) << endl;
    }
    if( transData.function_declaration.size()> 0) {
      for(int i = 0; i < transData.function_declaration.size(); i++) {
    	getline (file, line);
    	transData.cuda_typeFun.push_back(line);
        cout << transData.cuda_typeFun.at(i);
      }
    }
    file.close();
  }
}

/**
  * Translate from source program to targt program using a linear pattern.
  */
 void LinearTranslator::translateLinear(SgStatement* loopStat, SgLocatedNode *ln) {
   char *attStat;
  int location;
  string type;
  string var;
  string scalarParam;
  string varDeclaration;
  string allocation;
  string tmpCopy;
  string copyHD;
  string copyDH;
  string kernel;
  string size;
  string numThreads;
  string blocks;
  string nThreads;
  string tblock;
  string parameters;
 // string type;
  PreprocessingInfo * att;

  //read file
  readFile();

  att = attachArbitraryText(ln, " " ,  PreprocessingInfo::before);
  att = attachArbitraryText(ln, "  /***** Starting Parallalization *****/" ,  
    PreprocessingInfo::before);
  
  // ********************** declare device variables **************************/
  att = attachArbitraryText(ln,"  //declare device variables",  PreprocessingInfo::before);
  att = attachArbitraryText(ln,"  float elapsedTime;",  PreprocessingInfo::before);
  att = attachArbitraryText(ln,"  cudaEvent_t start, stop;",  PreprocessingInfo::before);
  att = attachArbitraryText(ln,"  cudaEventCreate(&start);",  PreprocessingInfo::before);
  att = attachArbitraryText(ln,"  cudaEventCreate(&stop);",  PreprocessingInfo::before);
  for(int i = 0; i < transData.device_variable.size(); i++) {
    location = transData.device_variable.at(i).find_first_of("[");
    var = transData.device_variable.at(i).substr(0, location);
    type = pairs[var];

    varDeclaration = type + " * " + " device_" + var + ";";
    attStat = new char[varDeclaration.size()+1];
	  strcpy(attStat, varDeclaration.c_str());
    att = attachArbitraryText(ln, attStat,  PreprocessingInfo::before);
  }
  att = attachArbitraryText(ln," ",  PreprocessingInfo::before);

  // ********************* allocate device variable ***************************/
  att = attachArbitraryText(ln,"  //Allocate memory space in the GPU",  
    PreprocessingInfo::before);
  for(int i = 0; i < transData.device_variable.size(); i++) {
    location = transData.device_variable.at(i).find_first_of("[");
    var = transData.device_variable.at(i).substr(0, location);
    varDeclaration = "device_" + var;
    allocation = "  cudaMalloc((void **) &" + varDeclaration + ", sizeof(" + var + "));";
    attStat = new char[allocation.size()+1];
	  strcpy(attStat, allocation.c_str());
    att = attachArbitraryText(ln, attStat,  PreprocessingInfo::before);
  }
  att = attachArbitraryText(ln," ",  PreprocessingInfo::before);

  // ********************* copy from host to device ***************************/
  att = attachArbitraryText(ln,"  //Copy from host to device", PreprocessingInfo::before);
  for(int i = 0; i < transData.device_variable.size(); i++) {
    if(transData.cuda_io.at(i) == "Input" || transData.cuda_io.at(i) == "Input/Output") {
     location = transData.device_variable.at(i).find_first_of("[");
     var = transData.device_variable.at(i).substr(0, location);
     varDeclaration = "device_" + var;
     copyHD = "  cudaMemcpy(" + varDeclaration + ", " + var + ", sizeof(" + var + "), cudaMemcpyHostToDevice);";
     attStat = new char[copyHD.size()+1];
	  strcpy(attStat, copyHD.c_str());
     att = attachArbitraryText(ln, attStat,  PreprocessingInfo::before);
   }
  }
  att = attachArbitraryText(ln," ",  PreprocessingInfo::before);
  
  // *********************** launch kernel ************************************/
  att = attachArbitraryText(ln,"  //launch kernel function", PreprocessingInfo::before);
  location = transData.num_of_loops.at(0).find_first_of("=");
  if(location > -1) {
	size = transData.num_of_loops.at(0).substr(2,transData.num_of_loops.at(0).size());
  } else {
  	size = transData.num_of_loops.at(0).substr(1, transData.num_of_loops.at(0).size());
  }
  numThreads = "dim3 numThreads(32,1);";
  blocks = "dim3 blocks((" + size + "+ 31)/32, 1);";
  nThreads = "numThreads";
  tblock = "blocks";
  parameters;
  
  att = attachArbitraryText(ln, "  " + numThreads,  PreprocessingInfo::before);
  att = attachArbitraryText(ln, "  " + blocks,  PreprocessingInfo::before);
  for(int i = 0; i < transData.device_variable.size(); i++) {
    location = transData.device_variable.at(i).find_first_of("[");
    var = transData.device_variable.at(i).substr(0, location);
    if(i < transData.device_variable.size()-1)
      varDeclaration = "device_" + var + ",";
    else 
       varDeclaration = "device_" + var;
    parameters += varDeclaration;
  }
  
  if(transData.scalar_variable.size() != 0) {
      if(transData.scalar_variable.size() == 1) {
        scalarParam = ", " + transData.scalar_variable.at(0);
      } else {
        for(int i = 0; i < transData.scalar_variable.size(); i++) {
           if(i == 0) {
             scalarParam = ", " + transData.scalar_variable.at(i) + ", ";
           }
           else if(i > 0 && i < transData.scalar_variable.size()-1) {
              scalarParam +=  transData.scalar_variable.at(i) + ",";
           } 
           else if(i > 0 && i < transData.scalar_variable.size()) {
              scalarParam += transData.scalar_variable.at(i);
           }
         }
        }
     }
     
     parameters += scalarParam;
  
  att = attachArbitraryText(ln,"  cudaEventRecord(start, 0);", PreprocessingInfo::before);
  kernel = "  kernel<<<" + tblock + "," + nThreads + ">>>(" + parameters + ");";
  attStat = new char[kernel.size()+1];
	strcpy(attStat, kernel.c_str());
  att = attachArbitraryText(ln, attStat,  PreprocessingInfo::before);
  att = attachArbitraryText(ln,"  cudaEventRecord(stop, 0);", PreprocessingInfo::before);
  att = attachArbitraryText(ln,"  cudaEventSynchronize(stop);", PreprocessingInfo::before);
  att = attachArbitraryText(ln,"  cudaEventElapsedTime(&elapsedTime, start, stop);", PreprocessingInfo::before);
  att = attachArbitraryText(ln,"  printf(\"the elapsed time is %f\\n\", elapsedTime);", PreprocessingInfo::before);
  att = attachArbitraryText(ln," ",  PreprocessingInfo::before);
  
  // ************************ copy back from device to host *******************/
  att = attachArbitraryText(ln,"  //copy back from device to host ",  
    PreprocessingInfo::before);
  for(int i = 0; i < transData.device_variable.size(); i++) {
    if(transData.cuda_io.at(i) == "Output" || transData.cuda_io.at(i) == "Input/Output") {
      location = transData.device_variable.at(i).find_first_of("[");
      var = transData.device_variable.at(i).substr(0, location);
      varDeclaration = "device_" + var;
      copyHD = "  cudaMemcpy(" + var + ", " + varDeclaration + ", sizeof(" + var + "), cudaMemcpyDeviceToHost);";
      attStat = new char[copyHD.size()+1];
	  strcpy(attStat, copyHD.c_str());
      att = attachArbitraryText(ln, attStat,  PreprocessingInfo::before);
    }
  }
  // ************************ free cuda memory *******************/
  for(int i = 0; i < transData.device_variable.size(); i++) {
    location = transData.device_variable.at(i).find_first_of("[");
    var = transData.device_variable.at(i).substr(0, location);
    varDeclaration = "device_" + var;
    att = attachArbitraryText(ln, "  cudaFree("+varDeclaration+");" ,  PreprocessingInfo::before);  
  }

att = attachArbitraryText(ln, " " ,  PreprocessingInfo::before);
  att = attachArbitraryText(ln, "  /***** Ending Parallalization *****/" , PreprocessingInfo::before);
  removeStatement(loopStat);
} 

