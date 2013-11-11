#ifndef DATA_H_
#define DATA_H_

/**
  * This header file has the data used by the translator.
  */
#include <vector>
using namespace std;

typedef struct
{
  string pattern;
  string parallel;
  string language;
  string offload;
  string nested;
  string operation;
  string sharedVar;
  string firstPriVar;
  string priVar;
  string redVar;
  string micVar;
  vector<string> cuda_io;
  vector<string> cuda_typeFun;
  vector<string> cuda_funDec;
  vector<string> type_variable;
  vector<string> scalar_variable;
  vector<string> loop_variable;
  vector<string> num_of_loops;
  vector<string> device_variable;
  vector<vector<string> > all_loops;
  vector<string> loop_file; 
  vector<string> function_declaration;
  vector<string> loopLine;
} TRANSDATA;

extern "C" TRANSDATA transData;
//extern TRANSDATA transData;
typedef map<string, string > MAPA;
//extern MAPA pairs;
extern "C" MAPA pairs;

#endif
