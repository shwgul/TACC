#include <iostream>
#include <string>
#include <fstream>
#include "Data.h"
#include "RemoverExtraLine.h"

using namespace std;

void RemoverExtraLine::removeExtraLine() {

  ifstream file ("rose_tomosm.c");
  ofstream file2 ("rose_tomosm.c");
  string line;

  if (file.is_open()) {
	while(!file.eof()) {
          getline (file, line);
	  for(int i = 0; i < transData.loop_file.size();i++) {
		if(transData.loop_file.at(i) != line) {
          		src_file.push_back(line);
       		}
	   }
        }
        file.close();
  }
  file2.open ("rose_tomosm.c");
  for(int i = 0; i < src_file.size(); i++)
  	file2 << src_file.at(i);
 
  file2.close();
}



