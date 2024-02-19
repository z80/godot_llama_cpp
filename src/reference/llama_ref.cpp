
#include "llama_ref.h"
#include "core/templates/vector.h"
//#include <vector>

#include "llama.h"

namespace Ign
{



static Vector<llama_token> llama_tokenize(
		const struct llama_model *model,
		const std::string &text,
		bool add_bos,
		bool special)
{
	// upper limit for the number of tokens
	int n_tokens = text.length();
	if (add_bos)
		n_tokens += 1;

	Vector<llama_token> result;
	result.resize(n_tokens);
	n_tokens = llama_tokenize( model, text.data(), text.length(), result.ptrw(), result.size(), add_bos, special );
	if (n_tokens < 0)
	{
		result.resize(-n_tokens);
		int check = llama_tokenize( model, text.data(), text.length(), result.ptrw(), result.size(), add_bos, special );
		GGML_ASSERT( check == -n_tokens );
	}
	else
	{
		result.resize(n_tokens);
	}
	return result;
}

static Vector<llama_token> llama_tokenize(
		const struct llama_context *ctx,
		const std::string &text,
		bool add_bos,
		bool special) {
	return llama_tokenize(llama_get_model(ctx), text, add_bos, special);
}

static void llama_batch_add(
		struct llama_batch & batch,
		llama_token id,
		llama_pos pos,
		const Vector<llama_seq_id> &seq_ids,
		bool logits)
{
	batch.token[batch.n_tokens] = id;
	batch.pos[batch.n_tokens] = pos;
	batch.n_seq_id[batch.n_tokens] = seq_ids.size();
	for (size_t i = 0; i < seq_ids.size(); ++i)
	{
		batch.seq_id[batch.n_tokens][i] = seq_ids[i];
	}
	batch.logits[batch.n_tokens] = logits;

	batch.n_tokens++;
}

static void llama_batch_clear( struct llama_batch & batch )
{
	batch.n_tokens = 0;
}

static String llama_token_to_piece( Vector<char> & result, const struct llama_context *ctx, llama_token token)
{
	const int n_tokens = llama_token_to_piece( llama_get_model(ctx), token, result.ptrw(), result.size() );
	if (n_tokens < 0)
	{
		result.resize(-n_tokens);
		int check = llama_token_to_piece( llama_get_model(ctx), token, result.ptrw(), result.size() );
		//GGML_ASSERT(check == -n_tokens);
	}
	else
	{
		result.resize(n_tokens);
	}

	return String( result.ptr(), result.size() );
}






class LlamaRef::PD
{
public:
	static const int n_len;

	//gpt_params params;
	llama_model * model;
	llama_context * ctx;
	llama_batch batch;
	bool batch_finished;

	Vector<llama_token> tokens_list;
	int n_ctx;
	int n_cur;

	Vector<llama_token_data> candidates;
	Vector<char> vector_piece;


	PD()
	{
		batch_finished = true;
		model = nullptr;
		ctx = nullptr;
		llama_backend_init();
		//llama_numa_init(params.numa);
	}

	~PD()
	{
		llama_backend_free();
	}

	bool load_model(const String & model_name, int threads_qty=1, int seed=42 )
	{
		const PackedByteArray name = model_name.to_utf8_buffer();
		const char * stri = (const char *)name.ptr();

		llama_model_params model_params = llama_model_default_params();
		model = llama_load_model_from_file( stri, model_params );

		const bool model_ok = (model != nullptr);
		if (!model_ok)
			return false;

		llama_context_params ctx_params = llama_context_default_params();

		ctx_params.seed = seed;
		ctx_params.n_ctx = 512;
		ctx_params.n_threads = threads_qty;
		ctx_params.n_threads_batch = threads_qty;

		ctx = llama_new_context_with_model(model, ctx_params);
		const bool ctx_ok = (ctx != nullptr);
		if (!ctx_ok)
			return false;

		return true;
	}

	bool start( const String& prompt )
	{
		const PackedByteArray prompt_array = prompt.to_utf8_buffer();
		const char * prompt_stri = (const char *)prompt_array.ptr();

		tokens_list = llama_tokenize( ctx, prompt_stri, false, false );

		n_ctx = llama_n_ctx(ctx);
		const int n_kv_req = tokens_list.size() + (n_len - tokens_list.size());

		//LOG_TEE("\n%s: n_len = %d, n_ctx = %d, n_kv_req = %d\n", __func__, n_len, n_ctx, n_kv_req);

		// make sure the KV cache is big enough to hold all the prompt and generated tokens
		if (n_kv_req > n_ctx)
		{
			//LOG_TEE("%s: error: n_kv_req > n_ctx, the required KV cache size is not big enough\n", __func__);
			//LOG_TEE("%s:        either reduce n_len or increase n_ctx\n", __func__);
			return false;
		}


		// create a llama_batch with size 512
		// we use this object to submit token data for decoding

		if (!batch_finished)
		{
			llama_batch_free(batch);
		}
		batch_finished = false;
		batch = llama_batch_init( 512, 0, 1 );

		// evaluate the initial prompt
		for ( size_t i = 0; i < tokens_list.size(); i++ )
		{
			llama_batch_add( batch, tokens_list[i], i, { 0 }, false );
		}
		// llama_decode will output logits only for the last token of the prompt
		batch.logits[batch.n_tokens - 1] = true;

		n_cur = batch.n_tokens;

		// main loop
		return true;
	}

	bool next( String & piece )
	{
		if (n_cur > n_len)
		{
			llama_batch_free( batch );
			batch_finished = true;
			return false;
		}

		// Evaluate next token of the model.
		if (llama_decode(ctx, batch) != 0)
		{
			print_line( "llama_decode() failed" );
			return false;
		}

		// sample the next token
		{
			auto n_vocab = llama_n_vocab( model );
			auto *logits = llama_get_logits_ith( ctx, batch.n_tokens - 1 );

			candidates.clear();
			for (llama_token token_id = 0; token_id < n_vocab; token_id++)
			{
				candidates.push_back( llama_token_data{ token_id, logits[token_id], 0.0f } );
			}

			llama_token_data_array candidates_p = { candidates.ptrw(), candidates.size(), false };

			// sample the most likely token
			const llama_token new_token_id = llama_sample_token_greedy( ctx, &candidates_p );

			// is it an end of stream?
			if (new_token_id == llama_token_eos(model) || n_cur >= n_len)
			{
				llama_batch_free(batch);
				batch_finished = true;
				return false;
			}
			else
			{
				//LOG_TEE("%s", llama_token_to_piece(ctx, new_token_id).c_str());
				piece = llama_token_to_piece( vector_piece, ctx, new_token_id);

				// prepare the next batch
				llama_batch_clear( batch );

				// push this new token for next evaluation
				llama_batch_add( batch, new_token_id, n_cur, { 0 }, true );
			}

		}

		n_cur += 1;

		return true;
	}

};

const int LlamaRef::PD::n_len = 512;






void LlamaRef::_bind_methods()
{
	ClassDB::bind_method( D_METHOD("load_model", "file_name"), &LlamaRef::load_model );
	ClassDB::bind_method( D_METHOD("start", "prompt"),         &LlamaRef::start );
	ClassDB::bind_method( D_METHOD("next"),                    &LlamaRef::next );

	//ADD_PROPERTY( PropertyInfo( Variant::STRING, "seed" ), "set_seed", "get_seed" );
}

LlamaRef::LlamaRef()
{
	pd = memnew(PD);
}

LlamaRef::~LlamaRef()
{
	memdelete(pd);
}

bool LlamaRef::load_model( const String & file_name )
{
	const bool ret = pd->load_model(file_name);
	return ret;
}

bool LlamaRef::start( const String & prompt )
{
	const bool ret = pd->start(prompt);
	return ret;
}

Variant LlamaRef::next()
{
	String piece;
	const bool ok = pd->next(piece);
	if (!ok)
		return Variant();

	Variant ret = piece;
	//Variant::construct_from_string(piece, ret);
	return ret;
}


}

