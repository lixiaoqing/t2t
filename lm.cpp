#include "lm.h"

struct ID_converter : public lm::EnumerateVocab 
{
	ID_converter(vector<lm::WordIndex>* out, Vocab* vocab) : sub_to_kenlm_id(out), UNK_ID(0),tgt_vocab(vocab) { sub_to_kenlm_id->clear(); }
	void Add(lm::WordIndex index, const StringPiece &str) 
	{
		const int ori_id = tgt_vocab->get_id(str.as_string());
		if (ori_id >= sub_to_kenlm_id->size())
		{
			sub_to_kenlm_id->resize(ori_id + 1, UNK_ID);
		}
		sub_to_kenlm_id->at(ori_id) = index;
	}
	vector<lm::WordIndex>* sub_to_kenlm_id;
	const lm::WordIndex UNK_ID;
	Vocab* tgt_vocab;
};

LanguageModel::LanguageModel(const string &lm_file, Vocab *tgt_vocab)
{
	ID_converter id_converter(&ori_to_kenlm_id,tgt_vocab);
	Config conf;
	conf.enumerate_vocab = &id_converter;
	kenlm = new Model(lm_file.c_str(), conf);
	EOS = convert_to_kenlm_id(tgt_vocab->get_id("</s>"));
	cout<<"load language model file "<<lm_file<<" over\n";
};

lm::WordIndex LanguageModel::convert_to_kenlm_id(int wid)
{
	if (wid >= ori_to_kenlm_id.size())
		return 0;
	else
		return ori_to_kenlm_id[wid];
}

double LanguageModel::cal_increased_lm_score(Cand* cand) 
{
	RuleScore<Model> rule_score(*kenlm,cand->lm_state);
	if ( cand->type == OOV || ( cand->type == NORMAL && cand->cands_of_nt_leaves.empty() ) )  // OOV候选或者由不含非终结符的规则生成的候选
	{
		for (const auto wid : cand->tgt_wids)
		{
			rule_score.Terminal( convert_to_kenlm_id(wid) );
		}
	}
	else if (cand->type == NORMAL)                                                            // 由含非终结符的规则生成的候选
	{
		TgtRule &applied_rule = cand->matched_tgt_rules->at(cand->rule_rank);
		size_t nt_idx = 0;
		for (size_t i=0; i<applied_rule.aligned_src_positions.size(); i++)
		{
			if (applied_rule.aligned_src_positions[i] == -1)
			{
				rule_score.Terminal( convert_to_kenlm_id(applied_rule.tgt_leaves[i]) );
			}
			else
			{
				rule_score.NonTerminal(cand->cands_of_nt_leaves[nt_idx][cand->cand_rank_vec[nt_idx]]->lm_state);
				nt_idx++;
			}
		}
	}
	else if (cand->type == GLUE)                                                              // glue候选
	{
		for (size_t nt_idx=0; nt_idx<cand->cands_of_nt_leaves.size(); nt_idx++)
		{
			rule_score.NonTerminal(cand->cands_of_nt_leaves[nt_idx][cand->cand_rank_vec[nt_idx]]->lm_state);
		}
	}
	double increased_lm_score = rule_score.Finish();
	cand->lm_state.ZeroRemaining();
	return increased_lm_score;
}

double LanguageModel::cal_final_increased_lm_score(Cand* cand) 
{
	ChartState cstate;
	RuleScore<Model> rule_score(*kenlm, cstate);
	rule_score.BeginSentence();
	rule_score.NonTerminal(cand->lm_state, 0.0f);
	rule_score.Terminal(EOS);
	return rule_score.Finish();
}
