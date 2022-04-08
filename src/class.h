#pragma once

class Timer{
	std::chrono::system_clock::time_point st, to;
	public:
	Timer(){
	}
	void start(){
		st = std::chrono::system_clock::now();
	}
	void stop(){
		to = std::chrono::system_clock::now();
	}
	void dumpWCTime(){
		std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(to - st).count() << " msec"  << std::endl;
	}
};

template <typename real> class mpicin{
	real value;
	public:
	mpicin(){
		if(PS::Comm::getRank() == 0){
			std::cin >> value;
		}
		PS::Comm::broadcast(&value, 1, 0);
	}
	const real getValue() const{
		return value;
	}
};

enum TYPE{
	HYDRO,
	FREEZE,
};

struct system_t{
	PS::F64 dt, time;
	PS::S64 step;
	system_t() : step(0), time(0.0), dt(0.0){
	}
};

class FileHeader{
public:
	int Nbody;
	double time;
	int readAscii(FILE* fp){
		fscanf(fp, "%.16lf\n", &time);
		fscanf(fp, "%d\n", &Nbody);
		return Nbody;
	}
	void writeAscii(FILE* fp) const{
		fprintf(fp, "%e\n", time);
		fprintf(fp, "%d\n", Nbody);
	}
};

namespace STD{
	namespace RESULT{
		//Hydro force
		class Hydro{
			public:
			PS::F64vec acc;
			PS::F64 dt;
			PS::F64 div_v;
			void clear(){
				acc = 0.0;
				dt = 1.0e+30;
				div_v = 0.0;
			}
		};
		//Shepard filter
		class Filter{
			public:
			PS::F64 dens;
			void clear(){
				dens = 0.0;
			}
		};
	}

	class RealPtcl{
		public:
		PS::F64 mass;
		PS::F64vec pos, vel, acc;
		PS::F64 dens;//DENSity
		PS::F64 pres;//PRESsure
		PS::F64 smth;//SMooTHing length
		PS::F64 snds; //SouND Speed
		PS::F64 div_v;

		PS::F64vec vel_half;
		PS::F64 dt;

		PS::S64 id, tag;

		const EoS::EoS_t<PS::F64>* EoS;
		const Viscosity::Viscosity_t<PS::F64>* visc;

		TYPE type;
		//Constructor
		RealPtcl(){
			type = HYDRO;
		}
		//Copy functions
		void copyFromForce(const RESULT::Hydro& force){
			this->acc   = force.acc;
			this->dt    = force.dt;
			this->div_v = force.div_v;
		}
		void copyFromForce(const RESULT::Filter& force){
			this->dens = force.dens;
		}
		//Give necessary values to FDPS
		PS::F64 getCharge() const{
			return this->mass;
		}
		PS::F64vec getPos() const{
			return this->pos;
		}
		PS::F64 getRSearch() const{
			return kernel_t::supportRadius() * this->smth;
		}
		void setPos(const PS::F64vec& pos){
			this->pos = pos;
		}
		void writeAscii(FILE* fp) const{
			#ifdef PARTICLE_SIMULATOR_TWO_DIMENSION
			fprintf(fp, "%ld\t%ld\t%.16lf\t%.16lf\t%.16lf\t%.16lf\t%.16lf\t%.16lf\t%.16lf\t%.16lf\t%.16lf\t%.16lf\n",  id,  tag,  mass,  pos.x,  pos.y,  0.0  ,  vel.x,  vel.y,  0.0  ,  dens,  snds,  pres);
			#else
			fprintf(fp, "%ld\t%ld\t%.16lf\t%.16lf\t%.16lf\t%.16lf\t%.16lf\t%.16lf\t%.16lf\t%.16lf\t%.16lf\t%.16lf\n",  id,  tag,  mass,  pos.x,  pos.y,  pos.z,  vel.x,  vel.y,  vel.z,  dens,  0.0,  pres);
			#endif
		}
		void readAscii(FILE* fp){
			#ifdef PARTICLE_SIMULATOR_TWO_DIMENSION
			fscanf (fp, "%ld\t%ld\t%.16lf\t%.16lf\t%.16lf\t%.16lf\t%.16lf\t%.16lf\t%.16lf\t%.16lf\t%.16lf\t%.16lf\n", &id, &tag, &mass, &pos.x, &pos.y, NULL, &vel.x, &vel.y, NULL, &dens, NULL, &pres);
			#else
			fscanf (fp, "%ld\t%ld\t%.16lf\t%.16lf\t%.16lf\t%.16lf\t%.16lf\t%.16lf\t%.16lf\t%.16lf\t%.16lf\t%.16lf\n", &id, &tag, &mass, &pos.x, &pos.y, &pos.z, &vel.x, &vel.y, &vel.z, &dens, NULL, &pres);
			#endif
		}
		void initialize(){
			smth = PARAM::SMTH * pow(mass / dens, 1.0/(PS::F64)(PS::DIMENSION));
		}
		void initialKick(const PS::F64 dt_glb){
			if(type == FREEZE) return;
			vel_half = vel + 0.5 * dt_glb * acc;
		}
		void fullDrift(const PS::F64 dt_glb){
			dens += - dens * dt_glb * div_v;
			//smth = PARAM::SMTH * pow(mass / dens, 1.0/(PS::F64)(PS::DIMENSION));
			if(type == FREEZE) return;
			pos += dt_glb * vel_half;
		}
		void predict(const PS::F64 dt_glb){
			if(type == FREEZE) return;
			vel += dt_glb * acc;
		}
		void finalKick(const PS::F64 dt_glb){
			if(type == FREEZE) return;
			vel = vel_half + 0.5 * dt_glb * acc;
		}
	};

	namespace EPI{
		class Hydro{
			public:
			PS::F64vec pos;
			PS::F64vec vel;
			PS::F64    smth;
			PS::F64    dens;
			PS::F64    pres;
			PS::F64    snds;
			PS::F64    visc;//kinetic viscosity
			PS::S64    tag;///DEBUG
			void copyFromFP(const RealPtcl& rp){
				this->pos  = rp.pos;
				this->vel  = rp.vel;
				this->smth = rp.smth;
				this->dens = rp.dens;
				this->pres = rp.pres;
				this->snds = rp.snds;
				this->tag  = rp.tag;///DEBUG
				this->visc = rp.visc->KineticViscosity();
			}
			PS::F64vec getPos() const{
				return this->pos;
			}
			PS::F64 getRSearch() const{
				return kernel_t::supportRadius() * this->smth;
			}
		};
		class Filter{
			public:
			PS::F64vec pos;
			PS::F64    smth;
			void copyFromFP(const RealPtcl& rp){
				this->pos  = rp.pos;
				this->smth = rp.smth;
			}
			PS::F64vec getPos() const{
				return this->pos;
			}
			PS::F64 getRSearch() const{
				return kernel_t::supportRadius() * this->smth;
			}
		};
	}

	namespace EPJ{
		class Hydro{
			public:
			PS::F64vec pos;
			PS::F64vec vel;
			PS::F64    dens;
			PS::F64    mass;
			PS::F64    smth;
			PS::F64    pres;
			PS::F64    snds;
			PS::F64    visc;//kinetic viscosity
			PS::S64    tag ;///DEBUG
			TYPE type;
			void copyFromFP(const RealPtcl& rp){
				this->pos  = rp.pos;
				this->vel  = rp.vel;
				this->dens = rp.dens;
				this->pres = rp.pres;
				this->smth = rp.smth;
				this->mass = rp.mass;
				this->snds = rp.snds;
				this->type = rp.type;
				this->tag  = rp.tag;
				this->visc = rp.visc->KineticViscosity();
			}
			PS::F64vec getPos() const{
				return this->pos;
			}
			PS::F64 getRSearch() const{
				return kernel_t::supportRadius() * this->smth;
			}
			void setPos(const PS::F64vec& pos){
				this->pos = pos;
			}
		};
		class Filter{
			public:
			PS::F64vec pos;
			PS::F64    dens;
			PS::F64    mass;
			void copyFromFP(const RealPtcl& rp){
				this->pos  = rp.pos;
				this->dens = rp.dens;
				this->mass = rp.mass;
			}
			PS::F64vec getPos() const{
				return this->pos;
			}
			void setPos(const PS::F64vec& pos){
				this->pos = pos;
			}
		};
	}
}


template <class Ptcl> class Problem{
	Problem(){
	}
	public:
	static void setupIC(PS::ParticleSystem<Ptcl>&, system_t&, PS::DomainInfo&){
	}
	static void addExternalForce(PS::ParticleSystem<Ptcl>&, system_t&){
	}
	static void postTimestepProcess(PS::ParticleSystem<Ptcl>&, system_t&){
	}
};

