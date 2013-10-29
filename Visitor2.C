#include <iostream>
#include <cstring>
#include <rose.h>
#include <stdio.h>
#include <fstream>
#include <vector>
#include <rewrite.h>
#include "LinearTranslator.h"
#include "TwoHostTranslator.h"
#include "Data.h"
#include "Function.h"
#include "FunctionTwo.h"
#include "OpenMPReduce.h"
#include "OpenMPDirective.h"
#include "ParserFunDecl.h"
#include <boost/algorithm/string.hpp>
#include <map>

using namespace SageInterface;
using namespace SageBuilder;
using namespace AstFromString;
using namespace std;
MAPA pairs;
TRANSDATA transData;

void loadLoopData() {
  string line;
  ifstream file ("../GenData/parserData.txt");

  if (file.is_open()) {
    while ( !file.eof() ) {
      getline (file,line);
      transData.loop_file.push_back(line);
    }
    file.close();
  }
  else cout << "Unable to open file";
}

void loadDataTranslator() {
  char * ptk;
  char *str;
  int isMatch;
  string line;
  ifstream file ("../GenData/genData.txt");
  if (file.is_open()) {
    //pattern
    getline(file, line);
    getline (file,line);
    transData.pattern = line;

    // parallel option
    getline(file, line);
    getline (file,line);
    getline(file, line);
    transData.parallel = line;

    // language option
    getline(file, line);
    getline (file,line);
    getline(file, line);
    transData.language = line;

    //type variables
    getline(file, line);
    getline (file,line);
    while("##scalar_variables##" != line) {
      getline(file, line);
      if(line != "#" && line != "##scalar_variables##")
        transData.type_variable.push_back(line);
    }

    //scalar variable
    while("##Device_variables##" != line) {
      getline(file, line);
      if(line != "#" && line != "##Device_variables##")
        transData.scalar_variable.push_back(line);
    }

    while("##Loop_variable##" != line) {
      getline(file, line);
      if(line != "#" && line != "##Loop_variable##")
        transData.device_variable.push_back(line);
    }

    while("##function_declaration##" != line) {
      getline(file, line);
      if(line != "#" && line != "##function_declaration##")
        transData.loop_variable.push_back(line);
    }

    //device variables
    while("##Number_Of_Loops##" != line) {
      getline(file, line);
      if(line != "#" && line != "##Number_Of_Loops##")
        transData.function_declaration.push_back(line);
    }

    // number of loops
    while("##End##" != line) {
      getline(file, line);
      if(line != "#" && line != "##End##")
        transData.num_of_loops.push_back(line);
    }
  }
  else cout << "Unable to open file";   
}

void setTypeVarMap()  {
  string line;
  for(int i = 0; i < transData.type_variable.size(); i++) {
    line = transData.type_variable.at(i);
    std::vector<std::string> strs;
    boost::split(strs,line,boost::is_any_of("\t "));
    pairs[strs[1]]=strs[0];
  }
  cout <<"Variable and types "<<endl;
  cout<<"***************************"<<endl;
  map<string,string>::iterator it = pairs.begin(); 
  while (it != pairs.end()) { 
    cout<<it->first<<" "<<it->second<<endl;
    it++;
  }
  cout<<"******************************"<<endl;
}

int main(int argc, char *argv[]) {
  ios::sync_with_stdio();
  ofstream file;
  loadLoopData();
  string selectedLoop;
  for(int i = 0; i < transData.loop_file.size(); i++) {
    selectedLoop += transData.loop_file.at(i);
  }
  boost::erase_all(selectedLoop," ");
  std::cout<<"Loop to parallelize without spaces "<<endl<<selectedLoop<<endl;
  if(SgProject::get_verbose() > 0) {
    cout << "In processor.c: main() \n";
  }
  SgProject* project = frontend(argc, argv);
  ROSE_ASSERT(project != NULL);
  //AstTests::runAllTests(const_cast<SgProject*>(project));
  if(project->get_verbose() > 1)
  {
    cout << AstNodeStatistics::traversalStatistics(project);
    cout << AstNodeStatistics::IRnodeUsageStatistics();
  }
  if(project->get_verbose() > 0) {
    cout<< "Generate Ouput "<<endl;
  }
  Rose_STL_Container<SgNode*> nodeList;
  nodeList = NodeQuery::querySubTree(project,  V_SgForStatement);
  Rose_STL_Container<SgNode*>::iterator nodeIterator = nodeList.begin();
  while(nodeIterator != nodeList.end())
  {
    string nodeString = (*nodeIterator)->unparseToString();
    boost::erase_all(nodeString," ");
    if(nodeString == selectedLoop) {
      file.open ("../GenData/log.txt");
      file << selectedLoop << " " << nodeString << endl;
      file.close();
      break;
    }
    nodeIterator++;
  }
  if (nodeIterator == nodeList.end()) {
    cout << "User input loop to be parallelized "
        <<" can't be found. Exiting with Error";
    return 0;
  }
  loadDataTranslator();
  setTypeVarMap();
  SgStatement* loopStat = (SgStatement *) (*nodeIterator);
  SgLocatedNode *locatedNode = isSgLocatedNode((*nodeIterator)); 
  LinearTranslator linearTranslator;
  TwoHostTranslator twoHostTranslator;  
  OpenMPReduce ompReducer;
  OpenMPDirective ompDirective;
  ParserFunDecl parserFn;

  if (transData.pattern == "CUDA:Outer-for-loop" && transData.parallel == "CUDA") {
    if (transData.language == "C" ||transData.language == "CPP") {
      linearTranslator.translateLinear(loopStat, locatedNode);
      nodeList = NodeQuery::querySubTree(project,  V_SgStatement);
      if (transData.function_declaration.size() > 0) {
        for (int j = 0; j < transData.function_declaration.size(); j++) {
          parserFn.parseFunDecl(nodeList, transData.function_declaration.at(j), j);
        }
      }
      SimpleInstrumentation treeTraversal;
      treeTraversal.traverseInputFiles(project, preorder);
    }
  } else if (transData.pattern == "CUDA:Double-nested-for-loop" 
      && transData.parallel == "CUDA") {
    if(transData.language == "C" ||transData.language == "CPP") {
      cout<<"Parallelizing Double nested loop"<<endl;
      twoHostTranslator.translateTwoHost(loopStat, locatedNode);
      nodeList = NodeQuery::querySubTree(project,  V_SgStatement);
      for(int j = 0; j < transData.function_declaration.size(); j++) {
        parserFn.parseFunDecl(nodeList, transData.function_declaration.at(j), j);
      }
      // create kernel function
      SimpleInstrumentationTwo treeTraversal;
      treeTraversal.traverseInputFiles(project, preorder);
    }
  } else if (transData.pattern == "OpenMP:Outer-for-loop, reduce with sum" 
      && transData.parallel == "OpenMP"){
    if(transData.language == "C") {
      ompReducer.translate(loopStat, locatedNode);
      nodeList = NodeQuery::querySubTree(project,  V_SgStatement);
      ompDirective.parseOpenMPDirective(nodeList);
      for(int j = 0; j < transData.function_declaration.size(); j++) {
        parserFn.parseFunDecl(nodeList, transData.function_declaration.at(j), j);    
      }
    } 
  }

  AstTests::runAllTests(const_cast<SgProject*>(project));
  backend(project);
  string outputFile = argv[argc-1];
  string cmd = "nvcc "+outputFile;
  cout<<"Compiling "<<outputFile<<endl;
  cout<<"Executing "<<cmd<<endl;
  cout<<"Compiler Output "<<endl;
  system(cmd.c_str());
  return 0;
}
