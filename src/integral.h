#pragma once
template <class ThisPtcl> PS::F64 getTimeStepGlobal(const PS::ParticleSystem<ThisPtcl>& sph_system){
	PS::F64 dt = 1.0e+30;//set VERY LARGE VALUE
	for(PS::S32 i = 0 ; i < sph_system.getNumberOfParticleLocal() ; ++ i){
		PS::F64 dt_tmp = 1.0e+30;
		if(sph_system[i].type == FREEZE) continue;
		dt_tmp = std::min(0.25 * sqrt(sph_system[i].smth / sqrt(sph_system[i].acc * sph_system[i].acc)), dt_tmp);
		dt_tmp = std::min(0.125 * sph_system[i].smth * sph_system[i].smth / sph_system[i].visc->KineticViscosity(), dt_tmp);
		dt = std::min(dt, std::min(sph_system[i].dt, dt_tmp));
	}
	return PS::Comm::getMinValue(dt);
}

