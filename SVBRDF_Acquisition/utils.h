#pragma once
#include <iostream>
#include <vector>
#include <cctype>
#include <string.h>
#include <time.h>
#include <fstream>

#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>

using namespace cv;
using namespace std;

class Uitls
{
public:
	Uitls();
	~Uitls();
	
	static bool readStringListFromYml(const string& filename, vector<string>& l);
	static bool readStringListFromText(const string& filename, vector<string> &l);
private:

};

