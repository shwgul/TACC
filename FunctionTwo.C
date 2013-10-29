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
  if ( globalScope != NULL) {
    //Create a parameter list
    SgFunctionParameterList* parameterList = buildFunctionParameterList();
    for(int i = 0; i < transData.device_variable.size(); i++) {
      string device_variable = transData.device_variable.at(i);
      int location = device_variable.find_first_of("[");
      bool is2D = false;
      if (device_variable.find_last_of("[")!=location) { 
        is2D = true;
      }
      device_variable = device_variable.substr(0,location);
      string type = pairs[device_variable];;
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

    string numLoops = transData.num_of_loops.at(0);
    var = transData.loop_variable.at(0);
    //TODO put conditions over all the loop variables
    string condition = "  if(" + var + " " + numLoops + "){";
    att = attachArbitraryText(ln, condition,  PreprocessingInfo::inside);

    for(int j = 1; j < transData.loop_file.size(); j++) {
      string line = transData.loop_file.at(j);
      att = attachArbitraryText(ln, line,  PreprocessingInfo::inside);
    }   
    prependStatement(function, globalScope);
  } 
}
