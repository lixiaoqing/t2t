#include "syntaxtree.h"

SyntaxTree::SyntaxTree(const string &line_of_tree)
{
	if (line_of_tree.size() > 3)
	{
		build_tree_from_str(line_of_tree);
		update_attrib(root);
	}
	else
	{
		root = NULL;
		sen_len = 0;
	}
	//dump(root);
}

void SyntaxTree::build_tree_from_str(const string &line_of_tree)
{
	vector<string> toks = Split(line_of_tree);
	SyntaxNode* cur_node;
	SyntaxNode* pre_node;
	int word_index = 0;
	for(size_t i=0;i<toks.size();i++)
	{
		//左括号情形，且去除"("作为终结符的特例(做终结符时，后继为“）”)
		if(toks[i]=="(" && i+1<toks.size() && toks[i+1]!=")")
		{
			string test=toks[i];
			if(i == 0)
			{
				root     = new SyntaxNode;
				pre_node = root;
				cur_node = root;
			}
			else
			{
				cur_node = new SyntaxNode;
				cur_node->father = pre_node;
				pre_node->children.push_back(cur_node);
				pre_node = cur_node;
			}
		}
		//右括号情形，去除右括号“）”做终结符的特例（做终结符时，前驱的前驱为“（,而且前驱不是")"
		else if(toks[i]==")" && !(i-2>=0 && toks[i-2] =="(" && toks[i-1] != ")"))
		{
			pre_node = pre_node->father;
			cur_node = pre_node;
		}
		//处理形如 （ VV 需要 ）其中VV节点这样的情形
		else if((i-1>=0 && toks[i-1]=="(") && (i+2<toks.size() && toks[i+2]==")"))
		{
			cur_node->label  = toks[i];
			cur_node         = new SyntaxNode;
			cur_node->father = pre_node;
			pre_node->children.push_back(cur_node);
		}
		//处理形如 VP （ VV 需要 ） VP这样的节点 或 需要 这样的节点
		else
		{
			cur_node->label = toks[i];
			//如果是“需要”的情形，则记录中文词的序号
			if(toks[i+1]==")")
			{
				cur_node->span_lbound = word_index;
				cur_node->span_rbound = word_index;
				words.push_back(toks[i]);
				word_index++;
			}
		}
	}
	sen_len = word_index;
}

void SyntaxTree::update_attrib(SyntaxNode* node)
{
	for (const auto child : node->children)
	{
		if (child->span_lbound == 9999)
		{
			update_attrib(child);
		}
		node->span_lbound = min(node->span_lbound,child->span_lbound);
		node->span_rbound = max(node->span_rbound,child->span_rbound);
		if ( node->children.empty() )
		{
			node->type = WORD;
		}
		else if ( node->span_lbound==node->span_rbound && node->children.size()==1 && node->children[0]->children.empty() )
		{
			node->type = POS;
		}
		else
		{
			node->type = CONSTITUENT;
		}
	}
	int key = ((node->span_lbound)<<16) + node->span_rbound;
	auto it = nodes_at_span.find(key);
	if ( it == nodes_at_span.end() )
	{
		vector<SyntaxNode*> nodes = {node};
		nodes_at_span.insert( make_pair(key,nodes) );
	}
	else
	{
		it->second.push_back(node);
	}
}

void SyntaxTree::dump(SyntaxNode* node)
{
	cout<<" ( "<<node->label<<' '<<node->span_lbound<<' '<<node->span_rbound<<' '<<node->type;
	for (const auto &child : node->children)
		dump(child);
	cout<<" ) ";
}
