extends Node


# Called when the node enters the scene tree for the first time.
func _ready():
	var model: LlamaCpp = LlamaCpp.new()
	var opened_ok: bool = model.load_model( "models/ggml-model-Q8_0.gguf" )
	print( opened_ok )


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta):
	pass
