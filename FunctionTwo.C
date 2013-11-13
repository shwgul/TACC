#include <iostream>
#include <rose.h>
#include <stdio.h>
#include <string>
#include <rewrite.h>
#include "LinearTranslator.h"
#include "Data.h"
#include "FunctionTwo.h"
#include <map>
#include <boost/algorithm/string.hpp>
using namespace SageInterface;
using namespace SageBuilder;
using namespace std;
struct Node { 
  string var; 
  string f_i; 
  string s_i;
  Node(string v, string f, string s) { 
    var = v; 
    f_i = f; 
    s_i = s;
  }
};
int ff(string line, string f) { 
  return line.find_first_of(f);
}
string sub(string line, int s,int e) { 
  return line.substr(s,e-s);
}
int fna(string line) { 
  return line.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ");
}
int lna(string line) { 
  return line.find_last_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ");
}
vector<Node> getNodes(string line) { 
  vector<Node> v;
  while(!line.empty() && line.find("[")!=string::npos) { 
    string f_index = sub(line, ff(line,"[")+1, ff(line,"]"));
    string variable = sub(line, lna(sub(line,0,ff(line,"[")))+1,ff(line,"["));
    line = line.substr(line.find_first_of("]")+1);
    string s_index = sub(line, ff(line,"[")+1, ff(line,"]"));
    v.push_back(Node(variable,f_index,s_index));
    line = line.substr(line.find_first_of("]")+1);
  }
  return v;
}
string tempS(Node node) { 
  return node.var+"["+node.f_i+"]["+node.s_i+"]";
}
string transformLine(string line,map<string,pair<string,pair<string,string> > > varMap) { 
  boost::trim_left(line);
  //cout<<line<<endl;
  vector<Node> ans = getNodes(line);
  string ret = "";
  for(int i=0;i<ans.size();i++) { 
    Node node = ans[i];
    pair<string,pair<string,string> > pp = varMap[node.var];
    //cout<<node.var<<endl;
    string sl = pp.second.second;
    string firstPart = line.substr(0,line.find(tempS(node)));
    string refNode =  node.var+"[("+node.f_i+")*("+sl+")+("+node.s_i+")]";;
    line = line.substr(line.find(tempS(node)) + tempS(node).size());
    ret+=(firstPart+refNode);
  }
  ret+=line;
  return ret;
}
/**
 * Creates the kernel function based on a linear patter.
 */
void SimpleInstrumentationTwo::visit(SgNode* astNode) {
  int location;
  string var, tmp, type;
  PreprocessingInfo * att;
  SgLocatedNode *ln;
  SgVariableDeclaration * varDeclaration, * varDeclaration2;
  SgGlobal* globalScope = isSgGlobal(astNode);
  if ( globalScope != NULL) {
    //Create a parameter list
    SgFunctionParameterList* parameterList = buildFunctionParameterList();
    map<string,pair<string,pair<string,string> > > varMap;
    for(int i = 0; i < transData.device_variable.size(); i++) {
      string device_variable = transData.device_variable.at(i);
      int location = device_variable.find_first_of("[");
      bool is2D = false;
      if (device_variable.find_last_of("[")!=location) { 
        is2D = false;
      }
      device_variable = device_variable.substr(0,location);
      string type = pairs[device_variable];;
      string ts = transData.device_variable.at(i);
      string fl = ts.substr(ts.find_first_of("[")+1,ts.find_first_of("]") - ts.find_first_of("[")-1);
      string sl = ts.substr(ts.find_last_of("[")+1,ts.find_last_of("]") - ts.find_last_of("[")-1);
      varMap[device_variable]=make_pair(type,make_pair(fl,sl));;
      SgName varName = device_variable;
      SgPointerType *ptrType = buildPointerType(buildIntType());
      if (type == "int") { 
        ptrType = buildPointerType(buildIntType());
      } else if (type == "float"){ 
        ptrType = buildPointerType(buildFloatType());
      } else if (type == "double") { 
        ptrType = buildPointerType(buildDoubleType()); 
      } else { 
        ptrType = buildPointerType(buildVoidType());
      }
      if (is2D) {
        ptrType = buildPointerType(ptrType);
      }  
      SgInitializedName *varIniName = buildInitializedName(varName, ptrType);
      appendArg(parameterList, varIniName);
    }

    for(int i = 0; i < transData.scalar_variable.size(); i++) {
      string scalar_variable = transData.scalar_variable.at(i);
      string type = pairs[scalar_variable];
      SgType *dataType = buildIntType();
      if (type == "int") { 
        dataType = buildIntType();
      } else if (type == "float"){ 
        dataType = buildFloatType();
      } else if (type == "double") { 
        dataType = buildDoubleType(); 
      } else { 
        dataType = buildVoidType();
      }
      SgName varName = scalar_variable;
      SgInitializedName *varIniName = buildInitializedName(varName, dataType);
      appendArg(parameterList, varIniName);
    }
    //Define function declaration
    SgName functionName = "__global__ kernel"; 
    SgFunctionDeclaration * function = buildDefiningFunctionDeclaration
      (functionName, buildVoidType(), parameterList, globalScope);
    SgBasicBlock *functionBody = function->get_definition()->get_body();
    function->set_global_qualification_required_for_return_type(true);    
    // create global function header
    ln = isSgLocatedNode((SgStatement *) functionBody);

    // create declaration for function
    /*TODO
      string tmp1;
      string tmp2;
      if(transData.cuda_funDec.size() > 0) {
      att = attachArbitraryText(ln, "#include<stdio.h>", PreprocessingInfo::before);
      for(int i = 0; i < transData.cuda_funDec.size(); i++) {
      tmp2 = transData.cuda_funDec.at(i);
      if(transData.cuda_typeFun.at(i) == "Device") tmp1 = "__device__";
      if(transData.cuda_typeFun.at(i) == "Device-Device") tmp1 = "__device__ __device__";
      if(transData.cuda_typeFun.at(i) == "Host-Device") tmp1 = "__host__ __device__";
      att = attachArbitraryText(ln, tmp1 + " " + tmp2, PreprocessingInfo::before);
      }
      }
      */

    //var = transData.loop_variable.at(0);

    //varDeclaration = buildVariableDeclaration(var, buildIntType()); 
    //ln = isSgLocatedNode(varDeclaration);
    // create local variables
    for(int i = 0; i < transData.loop_variable.size(); i++) {
      varDeclaration = buildVariableDeclaration(transData.loop_variable.at(i),buildIntType());
      prependStatement(varDeclaration,functionBody);
    } 

    //  create kernel function statements
    string thx = "  " + transData.loop_variable.at(0) + " = blockIdx.x * blockDim.x + threadIdx.x;";
    string thy = "  " + transData.loop_variable.at(1) + " = blockIdx.y * blockDim.y + threadIdx.y;";
    att = attachArbitraryText(ln, thx,  PreprocessingInfo::inside);
    att = attachArbitraryText(ln, thy,  PreprocessingInfo::inside);
    att = attachArbitraryText(ln, " ",  PreprocessingInfo::inside);
    string condition = "if ( ";
    for (int i = 0; i < transData.num_of_loops.size();i++) {
      string numLoops = transData.num_of_loops.at(i);
      var = transData.loop_variable.at(i);
      condition +=  var + " " + numLoops;
      if ( i!= transData.num_of_loops.size()-1) {
        condition += " && ";
      }
    }
    condition+="){";
    att = attachArbitraryText(ln, condition,  PreprocessingInfo::inside);
    for(int j = 2; j < transData.loop_file.size() - 2; j++) {
      string temp = transData.loop_file.at(j);
      string line = transformLine(temp,varMap);
      att = attachArbitraryText(ln, line,  PreprocessingInfo::inside);
    }   
    prependStatement(function, globalScope);
  } 
}
