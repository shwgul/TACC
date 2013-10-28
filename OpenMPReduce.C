#include <iostream>
#include <cstring>
#include <string>
#include <rose.h>
#include <stdio.h>
#include <vector>
#include <rewrite.h>
#include <fstream>
#include "OpenMPReduce.h"
#include "Data.h"
using namespace SageInterface;
using namespace SageBuilder;
using namespace AstFromString;
using namespace std;


/**
 *  Read files
 */
void OpenMPReduce::readFile1() {
  ifstream file ("../GenData/OpenMPReduceInData.txt");
  string line;	

  if (file.is_open()) {
    getline (file, line);
    transData.offload = line;
    getline (file, line);
    transData.operation = line;
    file.close();
  }
}

void OpenMPReduce::readFile2() {
  ifstream file("../GenData/OpenMPReduceOutData.txt");
  string line;

  if (file.is_open()) {
    getline (file, line);
    transData.sharedVar = line;
    getline (file, line);
    transData.firstPriVar = line;
    getline (file, line);
    transData.priVar = line;
    getline (file, line);
    transData.redVar = line;
    getline (file, line);
    transData.micVar = line;
    file.close();
  }  
}  

/**
  * Translate from source program to targt program using a linear pattern.
  */
void OpenMPReduce::translate(SgStatement* loopStat, SgLocatedNode *ln) {
  string op;
  string var;
  string shared_var;
  string priv_var;
  string firstPriv_var;
  string mic_var;
  int location1;
  int location2;
  PreprocessingInfo * att;

  // read files
  readFile1();
  readFile2();
 
  location1 = transData.sharedVar.find_first_of("[");
  location2 = transData.sharedVar.find_first_of("]");
  shared_var = transData.sharedVar.substr(location1+1, location2-1);

  location1 = transData.firstPriVar.find_first_of("[");
  location2 = transData.firstPriVar.find_first_of("]");
  firstPriv_var = transData.firstPriVar.substr(location1+1, location2-1);

  location1 = transData.priVar.find_first_of("[");
  location2 = transData.priVar.find_first_of("]");
  priv_var = transData.priVar.substr(location1+1, location2-1);

  location1 = transData.micVar.find_first_of("[");
  location2 = transData.micVar.find_first_of("]");
  mic_var = transData.micVar.substr(location1+1, location2-1);

  if(transData.operation == "Addition") op = "+";
  else if(transData.operation == "Subtration") op = "-";

  att = attachArbitraryText(ln, " " ,  PreprocessingInfo::before);
  att = attachArbitraryText(ln, "  /***** Starting Parallalization *****/" , PreprocessingInfo::before);

  if (transData.offload == "Yes") {
    att = attachArbitraryText(ln, "#pragma offload target (mic) in (" + mic_var + ") out(" + transData.redVar + ")" ,  PreprocessingInfo::before);
   }

    att = attachArbitraryText(ln,"#pragma omp parallel default(none) shared(" + shared_var + "," + transData.redVar + ") private (" + priv_var + ") firstprivate ( " + firstPriv_var + ")",  PreprocessingInfo::before);
    att = attachArbitraryText(ln, "{" ,  PreprocessingInfo::before);
    att = attachArbitraryText(ln, "#pragma omp for reduction (" + op + ":" + transData.redVar + ")" ,  PreprocessingInfo::before);
   
    for(int i = 0; i < transData.loop_file.size(); i++)
    	att = attachArbitraryText(ln, transData.loop_file.at(i), PreprocessingInfo::before);

    att = attachArbitraryText(ln, " printf(\"Inside :: Number of threads on target %d, solutions: %d \\n\", omp_get_num_threads(), solution_num);" ,  PreprocessingInfo::after);
    att = attachArbitraryText(ln, "}" ,  PreprocessingInfo::after);
    att = attachArbitraryText(ln, "  /***** Ending Parallalization *****/" , PreprocessingInfo::after);
    removeStatement(loopStat);
} 

