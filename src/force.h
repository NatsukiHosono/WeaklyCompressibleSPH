#pragma once
namespace STD{
	void CalcPressure(PS::ParticleSystem<STD::RealPtcl>& sph_system){
		#pragma omp parallel for
		for(PS::S32 i = 0 ; i < sph_system.getNumberOfParticleLocal() ; ++ i){
			sph_system[i].pres = sph_system[i].EoS->Pressure(sph_system[i].dens, 0.0);
			sph_system[i].snds = sph_system[i].EoS->SoundSpeed(sph_system[i].dens, 0.0);
		}
	}
	template <typename kernel_t> class ApplyFilter{
		public:
		void operator () (const EPI::Filter* const ep_i, const PS::S32 Nip, const EPJ::Filter* const ep_j, const PS::S32 Njp, RESULT::Filter* const filter){
			for(PS::S32 i = 0; i < Nip ; ++ i){
				PS::F64 one = 0.0;//should be one
				const EPI::Filter& ith = ep_i[i];
				for(PS::S32 j = 0; j < Njp ; ++ j){
					const EPJ::Filter& jth = ep_j[j];
					const PS::F64vec dr = ith.pos - jth.pos;
					const PS::F64 W = kernel_t::W(dr, ith.smth);
					one += W * jth.mass / jth.dens;
				}
				for(PS::S32 j = 0; j < Njp ; ++ j){
					const EPJ::Filter& jth = ep_j[j];
					const PS::F64vec dr = ith.pos - jth.pos;
					const PS::F64 W = kernel_t::W(dr, ith.smth);
					filter[i].dens += jth.mass * W / one;
				}
			}
		}
	};
	template <typename kernel_t> class CalcHydroForce{
		static constexpr double AV_STRENGTH = 0.01;
		public:
		void operator () (const EPI::Hydro* const ep_i, const PS::S32 Nip, const EPJ::Hydro* const ep_j, const PS::S32 Njp, RESULT::Hydro* const hydro){
			for(PS::S32 i = 0; i < Nip ; ++ i){
				PS::F64 v_sig_max = 0.0;
				const EPI::Hydro& ith = ep_i[i];
				for(PS::S32 j = 0; j < Njp ; ++ j){
					const EPJ::Hydro& jth = ep_j[j];
					const PS::F64vec dr = ith.pos - jth.pos;
					const PS::F64vec dv = ith.vel - jth.vel;
					if(dr * dr <= 0.0) continue;
					const PS::F64 c_ij = 0.5 * (ith.snds + jth.snds);
					const PS::F64 dens_ij = 0.5 * (ith.dens + jth.dens);
					const PS::F64 mu_ij = (dr * dv < 0.0) ? 0.5 * (ith.smth + jth.smth) * dv * dr / (dr * dr + 0.01 * ith.smth * jth.smth) : 0.0;
					const PS::F64 AV = - AV_STRENGTH * c_ij * mu_ij / dens_ij;
					v_sig_max = std::max(v_sig_max, mu_ij);
					//const PS::F64vec gradW = 0.5 * (kernel.gradW(dr, ith.smth) + kernel.gradW(dr, jth.smth));
					const PS::F64vec gradW = kernel_t::gradW(dr, 0.5 * (ith.smth + jth.smth));
					//pressure gradient
					hydro[i].acc -= jth.mass * (ith.pres / (ith.dens * ith.dens) + jth.pres / (jth.dens * jth.dens) + AV) * gradW;
					//viscosity
					hydro[i].acc += jth.mass * 4.0 * 0.5 * (ith.visc + jth.visc) * dr * gradW / (ith.dens + jth.dens) / (dr * dr + 0.01 * (ith.smth * jth.smth)) * dv;
					hydro[i].div_v += - jth.mass * dv * gradW;
				}
				hydro[i].dt = PARAM::C_CFL * ith.smth / (ith.snds + v_sig_max);
				hydro[i].div_v /= ith.dens;
			}
		}
	};
}


