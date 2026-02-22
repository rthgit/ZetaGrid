set ModuleHierarchy {[{
"Name" : "pack_int4_transpose","ID" : "0","Type" : "sequential",
"SubLoops" : [
	{"Name" : "block_loop","ID" : "1","Type" : "no",
	"SubInsts" : [
	{"Name" : "grp_pack_int4_transpose_Pipeline_read_loop_fu_254","ID" : "2","Type" : "sequential",
			"SubLoops" : [
			{"Name" : "read_loop","ID" : "3","Type" : "pipeline"},]},]},]
}]}