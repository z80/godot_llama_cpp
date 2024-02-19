
#ifndef __LLAMA_REF_H_
#define __LLAMA_REF_H_

#include <godot_cpp/classes/control.hpp>
#include <godot_cpp/classes/global_constants.hpp>
#include <godot_cpp/classes/image.hpp>
#include <godot_cpp/classes/input_event_key.hpp>
#include <godot_cpp/classes/tile_map.hpp>
#include <godot_cpp/classes/tile_set.hpp>
#include <godot_cpp/classes/viewport.hpp>
#include <godot_cpp/variant/variant.hpp>


#include <godot_cpp/core/binder_common.hpp>

namespace Llama
{

using namespace godot;

class LlamaCpp: public RefCounted
{
	GDCLASS(LlamaCpp, RefCounted);
protected:
	static void _bind_methods();

public:
	LlamaCpp();
	~LlamaCpp();

	bool load_model( const String & file_name );
	bool start( const String & prompt );
	Variant next();

private:
	// Hide all 3rd party includes inside of CPP file.
	class PD;
	PD * pd;
};


}



#endif

