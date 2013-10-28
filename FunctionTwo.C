#include <iostream>
#include <rose.h>
#include <stdio.h>
#include <string>
#include <rewrite.h>
#include "LinearTranslator.h"
#include "Data.h"
#include "FunctionTwo.h"

using namespace SageInterface;
using namespace SageBuilder;
using namespace std;


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
  if(globalScope != NULL) {
    //Create a parameter list
    SgName varName = "a";
    SgPointerType *ptrType = buildPointerType(buildFloatType());
    SgInitializedName *varIniName = buildInitializedName(varName, ptrType);
    SgFunctionParameterList* parameterList = buildFunctionParameterList();
    appendArg(parameterList, varIniName);
        
    //Define function declaration
    SgName functionName = "kernel";
    SgFunctionDeclaration * function = buildDefiningFunctionDeclaration
      (functionName, buildVoidType(), parameterList, globalScope);
    SgBasicBlock *functionBody = function->get_definition()->get_body();
    
    // Create a body
    
    // create parameter list
    string param, scalarParam;
    for(int i = 0; i < transData.device_variable.size(); i++) {
      location = transData.device_variable.at(i).find_first_of("[");
      tmp = transData.device_variable.at(i).substr(0, location);
      type = pairs[tmp];
      if(i < transData.device_variable.size()-1) {
        param += type + " * " + tmp + ",";
      } else {
        param += type + " * " + tmp;
      }
    }
    
    if(transData.scalar_variable.size() != 0) {
      if(transData.scalar_variable.size() == 1) {
         type = pairs[transData.scalar_variable.at(0)];
        scalarParam = ", " + type + " " + transData.scalar_variable.at(0);
      } else {
        for(int i = 0; i < transData.scalar_variable.size(); i++) {
           type = pairs[transData.scalar_variable.at(i)];
           if(i == 0) {
             scalarParam = ", " + type + " " + transData.scalar_variable.at(i) + ", ";
           }
           else if(i > 0 && i < transData.scalar_variable.size()-1) {
              scalarParam += type + " " + transData.scalar_variable.at(i) + ",";
           } 
           else if(i > 0 && i < transData.scalar_variable.size()) {
              scalarParam += type + " " + transData.scalar_variable.at(i);
           }
         }
        }
     }
        
    // create global function header
    ln = isSgLocatedNode((SgStatement *) functionBody);

    // create declaration for function
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

    if(scalarParam == " ") {
      att = attachArbitraryText(ln, "__global__ void kernel(" + param + ")", PreprocessingInfo::before);
    } else {
       att = attachArbitraryText(ln, "__global__ void kernel(" + param  + scalarParam+ ")", PreprocessingInfo::before);
    }
    
    var = transData.loop_variable.at(0);
    varDeclaration = buildVariableDeclaration(var, buildIntType()); 
    ln = isSgLocatedNode(varDeclaration);

    // create local variables
    if(transData.loop_variable.size() > 1) {
     for(int i = 1; i < transData.loop_variable.size(); i++) {
	att = attachArbitraryText(ln, "  int " + transData.loop_variable.at(i) + ";", PreprocessingInfo::before);
     }
    }
 
    //  create kernel function statements
    string thx = "  " + transData.loop_variable.at(0) + " = blockIdx.x * blockDim.x + threadIdx.x;";
    string thy = "  " + transData.loop_variable.at(1) + " = blockIdx.y * blockDim.y + threadIdx.y;";
    att = attachArbitraryText(ln, thx,  PreprocessingInfo::after);
    att = attachArbitraryText(ln, thy,  PreprocessingInfo::after);
    att = attachArbitraryText(ln, " ",  PreprocessingInfo::after);
    
    string numLoops = transData.num_of_loops.at(0);
    string condition = "  if(" + var + " " + numLoops + "){";
    att = attachArbitraryText(ln, condition,  PreprocessingInfo::after);
    
    for(int j = 1; j < transData.loop_file.size(); j++) {
      string line = transData.loop_file.at(j);
      att = attachArbitraryText(ln, line,  PreprocessingInfo::after);
    }   
 
    prependStatement(varDeclaration, functionBody);
    prependStatement(function, globalScope);
  } 
}
