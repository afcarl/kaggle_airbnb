#include <stdlib.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <set>
 
 
std::string Trim(std::string& str)
{
    str.erase(0,str.find_first_not_of(" \t\r\n"));
    str.erase(str.find_last_not_of(" \t\r\n") + 1);
    return str;
}

typedef std::map<std::string,int> FeatStatisticsMap;
typedef std::map<std::string,FeatStatisticsMap> IdFeatMap;
typedef std::set<std::string> FeatSet;

int main()
{
    std::ifstream fin("data/sessions_nonan.csv");
    std::string line;    
    IdFeatMap id_feat_map;
    FeatSet feat_sets;
    int line_index = 0;
    while (getline(fin, line)) 
    {
        std::cout<<line_index++<<std::endl;
        std::istringstream sin(line);    
        std::vector<std::string> fields;    
        std::string field;
        while (getline(sin, field, ',')) 
        {
            fields.push_back(field);    
        }
        std::string user_id = Trim(fields[0]);  
        std::string feat[5];
        feat[0] = Trim(fields[1]);   
        feat[1] = Trim(fields[2]);  
        feat[2] = Trim(fields[3]);  
        feat[3] = Trim(fields[4]);  
        feat[4] = Trim(fields[5]);
        std::stringstream ss;
        ss.str("");
        ss<<"mainaction_"<<feat[0];
        feat[0] = ss.str();
        ss.str("");
        ss<<"actiontype_"<<feat[1];
        feat[1] = ss.str();
        ss.str("");
        ss<<"actiondetail_"<<feat[2];
        feat[2] = ss.str();
        ss.str("");
        ss<<"devicetype_"<<feat[3];
        feat[3] = ss.str();
        ss.str("");
        IdFeatMap::iterator itr = id_feat_map.find(user_id); 
        //std::cout<<"user_id: "<<user_id<<std::endl;
        if(itr==id_feat_map.end())
        {
            FeatStatisticsMap tmp_feat_map;
            tmp_feat_map[feat[0]] = atoi(feat[4].c_str());
            tmp_feat_map[feat[1]] = atoi(feat[4].c_str());
            tmp_feat_map[feat[2]] = atoi(feat[4].c_str());
            tmp_feat_map[feat[3]] = atoi(feat[4].c_str());
            id_feat_map[user_id] = tmp_feat_map;
        }
        else
        {
            for(int i=0;i<4;i++)
            {
                feat_sets.insert(feat[i]);
                FeatStatisticsMap::iterator sub_itr = itr->second.find(feat[i]);
                if(sub_itr==itr->second.end())
                {
                    itr->second[feat[i]] = atoi(feat[4].c_str());
                }
                else
                {
                    sub_itr->second += atoi(feat[4].c_str());
                }
            }
        }
    }
    std::ofstream outfile("data/session_feat.csv",std::ios::out);
    outfile<<"user_id";
    for(FeatSet::iterator set_itr=feat_sets.begin();set_itr!=feat_sets.end();set_itr++)
    {
        outfile<<","<<*set_itr;
    }
    outfile<<std::endl;

    line_index = 0;
    int id_num = id_feat_map.size();
    for(IdFeatMap::iterator itr = id_feat_map.begin();itr!=id_feat_map.end();itr++)
    {
        std::cout<<"to csv:"<<line_index++<<"/"<<id_num<<std::endl;
        outfile<<itr->first;
        for(FeatSet::iterator set_itr=feat_sets.begin();set_itr!=feat_sets.end();set_itr++)
        {
            FeatStatisticsMap::iterator feat_itr = itr->second.find(*set_itr);
            if(feat_itr!=itr->second.end())
            {
                outfile<<","<<feat_itr->second;
            }
            else
            {
                outfile<<",0";
            }
        }
        outfile<<std::endl;
    }
    return 0;
}












