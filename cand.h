#ifndef CAND_H
#define CAND_H
#include "stdafx.h"
#include "ruletable.h"
#include "lm/left.hh"

//存储翻译候选
struct Cand	                
{
	//目标端信息
	int tgt_root;               //当前候选目标端的根节点
	vector<int> tgt_wids;		//当前候选目标端的id序列

	//打分信息
	double score;				//当前候选的总得分
	vector<double> trans_probs;	//翻译概率
	double lm_prob;

	//来源信息, 记录候选是如何生成的
	CandType type;                                 // 候选的类型(1.由OOV生成; 2.由普通规则生成; 3.由glue规则生成)
	string syntax_node_info;                       // 当前候选所对应的句法节点, 输出规则信息时用
	RuleTrieNode* rule_node;                       // 生成当前候选的规则的源端
	vector<TgtRule>* matched_tgt_rules;            // 目标端非终结符相同的一组规则
	int rule_rank;                                 // 当前候选所用的规则在matched_tgt_rules中的排名
	vector<vector<Cand*> > cands_of_nt_leaves;     // 规则源端非终结符叶节点的翻译候选(glue规则所有叶节点均为非终结符)
	vector<int> cand_rank_vec;                     // 记录当前候选所用的每个非终结符叶节点的翻译候选的排名
	vector<int> tgt_root_of_leaf_cands;            // 记录源端非终结符叶节点的翻译候选的目标端根节点
	int rule_num;                                  // 使用的规则的数量
	int grule_num;                                 // 使用的glue规则的数量
	int crule_num;                                 // 使用的compose规则的数量

	//语言模型状态信息
	lm::ngram::ChartState lm_state;

	Cand ()
	{
		tgt_root = -1;
		tgt_wids.clear();

		score = 0.0;
		trans_probs.resize(PROB_NUM,0);
		lm_prob = 0.0;

		type = INIT;
		rule_node = NULL;
		matched_tgt_rules = NULL;
		rule_rank = 0;
		cands_of_nt_leaves.clear();
		cand_rank_vec.clear();
		tgt_root_of_leaf_cands.clear();
		rule_num  = 0;
		grule_num = 0;
		crule_num = 0;
	}
	~Cand ()
	{
		tgt_root = -1;
		tgt_wids.resize(0);

		score = 0.0;
		trans_probs.resize(0);
		lm_prob = 0.0;

		type = INIT;
		rule_node = NULL;
		matched_tgt_rules = NULL;
		rule_rank = 0;
		cands_of_nt_leaves.resize(0);
		cand_rank_vec.resize(0);
		tgt_root_of_leaf_cands.resize(0);
		rule_num  = 0;
		grule_num = 0;
		crule_num = 0;
	}
};

struct smaller
{
	bool operator() ( const Cand *pl, const Cand *pr )
	{
		return pl->score < pr->score;
	}
};

bool larger( const Cand *pl, const Cand *pr );

//组织每个句法节点翻译候选的类
class CandOrganizer
{
	public:
		~CandOrganizer()
		{
			for (auto cand : all_cands)
			{
				delete cand;
			}
			for (auto cand : recombined_cands)
			{
				delete cand;
			}
		}
		bool add(Cand *&cand_ptr);
		void sort_and_group_cands();
	private:
		bool is_bound_same(const Cand *a, const Cand *b);

	public:
		vector<Cand*> all_cands;                         // 当前节点所有的翻译候选
		vector<Cand*> recombined_cands;                  // 被重组的翻译候选, 回溯查看所用规则的时候使用
		map<int,vector<Cand*> > tgt_root_to_cand_group;  // 将当前节点的翻译候选按照目标端的根节点进行分组
		map<string,int> recombine_info_to_cand_idx;      // 根据重组信息(由边界词与目标端根节点组成)找候选在all_cands中的序号
	                                                     // 以查找新候选是否跟已有候选重复, 如有重复则进行假设重组
};

typedef priority_queue<Cand*, vector<Cand*>, smaller> Candpq;

#endif
