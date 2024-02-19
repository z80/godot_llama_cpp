
#ifndef __LLAMA_REF_H_
#define __LLAMA_REF_H_

//#include "llama.h"
#include "core/object/ref_counted.h"


namespace Ign
{

class LlamaRef: public RefCounted
{
	GDCLASS(LlamaRef, RefCounted);
protected:
	static void _bind_methods();

public:
	LlamaRef();
	~LlamaRef();

	bool load_model( const String & file_name );
	bool start( const String & prompt );
	Variant next();

private:
	//IgnRandom rand;
	class PD;
	PD * pd;
};


}



#endif

