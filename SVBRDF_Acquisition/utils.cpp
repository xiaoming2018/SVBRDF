#include "utils.h"
Uitls::Uitls()
{
}

Uitls::~Uitls()
{
}

bool Uitls::readStringListFromYml(const string& filename, vector<string>& l)
{
	l.resize(0);
	cv::FileStorage fs(filename, cv::FileStorage::READ);
	if (!fs.isOpened())
		return false;
	cv::FileNode n = fs.getFirstTopLevelNode();
	if (n.type() != cv::FileNode::SEQ)
		return false;
	FileNodeIterator itor = n.begin(), itor_end = n.end();
	for (; itor != itor_end; itor++)
	{
		std::string temp = (std::string)*itor;
		l.push_back((std::string)*itor);
	}
	return true;
}

bool Uitls::readStringListFromText(const string& filename, vector<string>& l)
{
	l.resize(0);
	ifstream input(filename, ifstream::in);
	if (!input.is_open()) {
		std::cout << "open file falied " << std::endl;
		return false;
	} else {
		while (!input.eof())
		{
			std::string buffer;
			getline(input, buffer);
			l.push_back(buffer);
		}
	}
	input.close();
	return true;
}
