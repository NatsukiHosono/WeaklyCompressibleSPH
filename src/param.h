#pragma once

namespace PARAM{
	const PS::F64 SMTH = 1.2;
	const PS::F64 C_CFL = 0.3;
	const PS::U64 NUMBER_OF_SNAPSHOTS = 1560;
	//Tree parameters
	const PS::U32 Nleaf  = 8;
	const PS::U32 Ngroup = 128;
	//neighbour list reusing timestep
	const PS::U32 REUSE_LIST_INTERVAL = 1;
	const PS::U32 FILTER_APPLYING_INTERVAL = 30;
};

