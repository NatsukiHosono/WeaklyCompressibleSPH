#pragma once

namespace Viscosity{
	//////////////////
	//abstract classes
	//////////////////
	template <typename type> class YieldCriterion_t{
		public:
		YieldCriterion_t(){
			return ;
		}
		virtual ~YieldCriterion_t(){
			return ;
		}
	};
	template <typename type> class Viscosity_t{
		public:
		Viscosity_t(){
			return ;
		}
		virtual ~Viscosity_t(){
			return ;
		}
		virtual type KineticViscosity() const = 0;
	};
	//////////////////
	//concrete classes
	//////////////////
	template <typename type> class Newtonian : public Viscosity_t<type>{
		const type nu0;
		public:
		Newtonian(const type _nu0) : nu0(_nu0){
		}
		type KineticViscosity() const{
			return nu0;
		}
	};
	static const Viscosity::Newtonian<PS::F64> Water(1.0e-6);
	static const Viscosity::Newtonian<PS::F64> No(0.0);
}

namespace EoS{
	//////////////////
	//abstract class
	//////////////////
	template <typename type> class EoS_t{
		public:
		EoS_t(){
			return ;
		}
		virtual ~EoS_t(){
			return ;
		}
		virtual type Pressure  (const type dens, const type eng) const = 0;
		virtual type SoundSpeed(const type dens, const type eng) const = 0;
	};
	//////////////////
	//EoSs
	//////////////////
	template <typename type> class IdealGas : public EoS_t<type>{
		const type hcr;//heat capacity ratio;
		public:
		IdealGas(const type _hcr) : hcr(_hcr){
		}
		inline type Pressure(const type dens, const type eng) const{
			return (hcr - 1.0) * dens * eng;
		}
		inline type SoundSpeed(const type dens, const type eng) const{
			return sqrt(hcr * (hcr - 1.0) * eng);
		}
		inline type HeatCapacityRatio() const{
			return hcr;
		}
	};
	template <typename real> class Murnaghan : public EoS_t<real>{
		const real gmm, rho0, B; //, mu0; //
		public:
		Murnaghan(const real _gmm, const real _rho0, const real _c0) : gmm(_gmm), rho0(_rho0), B(_c0 * _c0 * _rho0 / _gmm){
		}
		inline real Pressure(const real dens, const real eng) const{
			//return B * (pow(dens / rho0, gmm) - 1.0);
			return std::max(B * (pow(dens / rho0, gmm) - 1.0), 0.0);
		}
		inline real SoundSpeed(const real dens, const real eng) const{
			return sqrt(std::max(B * gmm / rho0 * pow(dens / rho0, gmm - 1.0), 1.0e-16));
		}
		inline real ReferenceDensity(void) const{
			return rho0;
		}
	};
}

