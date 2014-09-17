#ifndef SYNTAXTREE_H
#define SYNTAXTREE_H
#include "stdafx.h"
#include "myutils.h"
#include "cand.h"
#include "vocab.h"

// 源端句法树节点
struct SyntaxNode
{
	string label;                                    // 该节点的句法标签或者词
	SyntaxNode* father;
	vector<SyntaxNode*> children;
	int span_lbound;                                 // 该节点对应的span的左边界
	int span_rbound;                                 // 该节点对应的span的右边界
	NodeType type;                                   // 该节点的类型, 可为 1.单词节点; 2.词性节点; 3.句法节点
	CandOrganizer cand_organizer;                    // 组织该节点的翻译候选
	
	SyntaxNode ()
	{
		father      = NULL;
		span_lbound = 9999;
		span_rbound = -1;
		type        = WORD;
	}
	~SyntaxNode ()
	{
		for (auto node : children)
		{
			delete node;
		}
	}
};

class SyntaxTree
{
	public:
		SyntaxTree(const string &line_of_tree);
		~SyntaxTree()
		{
			delete root;
		}

	private:
		void build_tree_from_str(const string &line_of_tree);
		void update_attrib(SyntaxNode* node);
		void dump(SyntaxNode* node);

	public:
		SyntaxNode* root;
		int sen_len;
		vector<string> words;
		map<int,vector<SyntaxNode*> > nodes_at_span;    // 记录每个跨度对应的所有节点
};

#endif
