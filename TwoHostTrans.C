#include <iostream>
#include <cstring>
#include <string>
#include <rose.h>
//#include <stdio.h>
#include <fstream>
#include <vector>
#include <rewrite.h>
#include "TwoHostTranslator.h"
#include "Data.h"

using namespace SageInterface;
using namespace SageBuilder;
using namespace AstFromString;
using namespace std;

void TwoHostTranslator::readFile() {
  ifstream file ("../GenData/cudaInData.txt");
  string line;

  if (file.is_open()) {
    for(int i = 0; i < transData.device_variable.size(); i++) {
      getline (file, line);
      transData.cuda_io.push_back(line);
    }
    if( transData.function_declaration.size()> 0) {
      for(int i = 0; i < transData.function_declaration.size(); i++) {
        getline (file, line);
        transData.cuda_typeFun.push_back(line);
      }
    }
    file.close();
  }
}

void TwoHostTranslator::setInnerLoop() {
  ifstream file ("../GenData/loops.txt");
  string line;
  if (file.is_open()) {
    while(!file.eof()) {
      getline (file, line);
      transData.loopLine.push_back(line);
    }
  }
  file.close();
/* This behavior is wrong 
 * TODO
  int counter = 0;
  int loc = -1;
  string var = transData.loop_variable.at(1);
  string upper = transData.num_of_loops.at(1);
  string clossBracks = "{";
  for(int i = 0; i <  transData.loop_file.size(); i++) {
    loc = transData.loop_file.at(i).find_first_of("{");
    for(int j = 0; j < transData.loopLine.size(); j++) {
      if(transData.loopLine.at(j) == transData.loop_file.at(i) && counter < 2) {
        if(loc > -1) {
          transData.loop_file.at(i) = "if("+var + upper+"){";
          loc = -1;
        } else {
          transData.loop_file.at(i) = "if("+var + upper+")";
        }
        counter++;
      }
    }
  }
  */
}

/**
 * Translate from source program to targt program using a linear pattern.
 */
void TwoHostTranslator::translateTwoHost(SgStatement* loopStat, SgLocatedNode *ln) {
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
  string size2;
  string numThreads;
  string blocks;
  string nThreads;
  string tblock;
  string parameters;
  // string type;
  PreprocessingInfo * att;

  //read file
  readFile();

  //find for loop
  setInnerLoop();
  cout<<"Read the file and inner loop";
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
    bool is2D = false;
    if (transData.device_variable.at(i).find_last_of("[") != location) { 
      is2D = true;
    }
    var = transData.device_variable.at(i).substr(0, location);
    type = pairs[var];
    if (is2D) {
      varDeclaration = type + " ** " + " device_" + var + ";";
    } else { 
      varDeclaration = type + " * " + " device_" + var + ";";
    }
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

  location = transData.num_of_loops.at(1).find_first_of("=");
  if(location > -1) {
    size2 = transData.num_of_loops.at(1).substr(2,transData.num_of_loops.at(1).size());
  } else {
    size2 = transData.num_of_loops.at(1).substr(1,transData.num_of_loops.at(1).size());
  }

  numThreads = "dim3 numThreads(32,32);";
  blocks = "dim3 blocks((" + size + "+ 31)/32, (" + size2 + "+ 31)/32);";
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

  ifstream file ("../GUI/rose_tomosm.c");
  string line;
  if (file.is_open()) {
    while(!file.eof()) {
      getline (file, line);
      src_file.push_back(line);
    }
  }
  file.close();

  for(int i = 0; i < src_file.size(); i++) {
    for(int j = 0; j < transData.loop_file.size(); j++) {
      if(src_file.at(i) != transData.loop_file.at(j)) {
        break;
      }
    }
    cout << src_file.at(i) << endl;
  }
} 

