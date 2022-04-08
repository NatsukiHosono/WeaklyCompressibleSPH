#pragma once

template <class Ptcl> class Dam: public Problem<Ptcl>{
	public:
	static constexpr double END_TIME = 6.5;

	static constexpr PS::F64 dx = 1.0 / 20.;//146.0 ;//92.0; //39.0;
	static constexpr PS::F64 wall = 0.1;
	static constexpr PS::F64 box_x = 3.22;
	static constexpr PS::F64 box_y = 1.0;
	static constexpr PS::F64 box_z = 1.0;

	//put a sensor
	static constexpr PS::F64 offset_x0_sensor = 0.6635;
	static constexpr PS::F64 offset_y0_sensor = 0.295;
	static constexpr PS::F64 size_x_sensor = 0.161;
	static constexpr PS::F64 size_y_sensor = 0.403;
	static constexpr PS::F64 size_z_sensor = 0.161;

	//water
	static constexpr PS::F64 offset_x0_water = offset_x0_sensor + size_x_sensor + 1.1675;
	static constexpr PS::F64 offset_y0_water = 0.0;
	static constexpr PS::F64 size_x_water = 1.228;
	static constexpr PS::F64 size_y_water = 1.0;
	static constexpr PS::F64 size_z_water = 0.55;
	static void setupIC(PS::ParticleSystem<Ptcl>& sph_system, system_t& sysinfo, PS::DomainInfo& dinfo){
		/////////
		//place ptcls
		/////////
		std::vector<Ptcl> ptcl;
		// set for dam break of http://www.math.rug.nl/~veldman/comflo/dambreak.html

		const std::size_t cnt = size_x_water * size_y_water * size_z_water / (dx * dx * dx);
		const PS::F64 mass_water = EoS::Water.ReferenceDensity() * size_x_water * size_y_water * size_z_water;
		const PS::F64 mass = mass_water / cnt;
		std::cout << "# of water: " << cnt << std::endl;
		std::cout << "mass of water: " << mass_water << std::endl;
		std::cout << " -> mass of ptcl: " << mass << std::endl;
		
		std::size_t i = 0;
		for(PS::F64 x = - wall ; x <= box_x + wall ; x += dx){
			for(PS::F64 y = - wall ; y <= box_y + wall ; y += dx){
				for(PS::F64 z = - wall ; z <= box_z + wall ; z += dx){
					++ i;
					Ptcl ith;
					if(x < 0.0 || y < 0.0 || z < 0.0 || x > box_x || y > box_y){
						//set for boundary
						ith.type = FREEZE;
						ith.tag = 1;
					}else if(offset_x0_sensor <= x && x <= offset_x0_sensor + size_x_sensor && z <= size_z_sensor && offset_y0_sensor <= y && y <= offset_y0_sensor + size_y_sensor){
						ith.type = FREEZE;
						ith.tag = 2;
					}else if(offset_x0_water - wall <= x && x < offset_x0_water && offset_y0_water <= y && y <= offset_y0_water + size_y_water){
						ith.type = FREEZE;
						ith.tag = 3;
					}else if(offset_x0_water  <= x && x <= offset_x0_water  + size_x_water  && z <= size_z_water  && offset_y0_water  <= y && y <= offset_y0_water  + size_y_water ){
						continue;
					}else{
						continue;
					}
					ith.pos.x = x;
					ith.pos.y = y;
					ith.pos.z = z;
					ith.dens = EoS::Water.ReferenceDensity();
					ith.mass = mass;
					ith.id = i;
					ith.EoS = &EoS::Water;
					ith.visc = &Viscosity::Water;
					ptcl.push_back(ith);
				}
			}
		}

		std::cout << "# of ptcls is... " << ptcl.size() << std::endl;
		if(PS::Comm::getRank() == 0){
			const PS::S32 numPtclLocal = ptcl.size();
			sph_system.setNumberOfParticleLocal(numPtclLocal);
			for(PS::U32 i = 0 ; i < ptcl.size() ; ++ i){
				sph_system[i] = ptcl[i];
			}
		}else{
			sph_system.setNumberOfParticleLocal(0);
		}
		//Fin.
		std::cout << "setup..." << std::endl;
	}
	static void addExternalForce(PS::ParticleSystem<Ptcl>& sph_system, system_t& system){
		for(PS::U64 i = 0 ; i < sph_system.getNumberOfParticleLocal() ; ++ i){
			sph_system[i].acc.z -= 9.8;
			//if(system.dt != 0.0) sph_system[i].acc += - 0.1 * sph_system[i].vel / system.dt;
		}
	}
	static void postTimestepProcess(PS::ParticleSystem<Ptcl>& sph_system, system_t& sysinfo){
		for(PS::U64 i = 0 ; i < sph_system.getNumberOfParticleLocal() ; ++ i){
			sph_system[i].dens = std::max(sph_system[i].dens, 100.);
		}
		if(PS::Comm::getRank() != 0) return;
		static PS::F64 mass = 0.0;
		static PS::F64 time = 0.0;
		//injection velocity
		const PS::F64vec velocity(3.0, 0.0, 0.0);
		//injection point
		const PS::F64vec point(offset_x0_water, 0.5 * box_y, 1.2 * box_z);
		//injection pipe square size
		const PS::F64 size = 0.2;
		//injection volume
		const PS::F64 volume = size_x_water * size_y_water * size_z_water;

		//injecting mass
		const PS::F64 total_mass = EoS::Water.ReferenceDensity() * volume;
		if(mass > total_mass) return;
		if(sysinfo.time >= time){
			for(double y = - size * 0.5 ; y <= size * 0.5 ; y += dx){
				for(double z = - size * 0.5 ; z <= size * 0.5 ; z += dx){
					Ptcl ith;
					ith.pos = PS::F64vec(0.0, y, z) + point;
					ith.vel.x = velocity.x;
					ith.acc.z = -9.8;
					ith.type = HYDRO;
					ith.tag = 0;
					ith.EoS = &EoS::Water;
					ith.visc = &Viscosity::Water;
					ith.dens = EoS::Water.ReferenceDensity();
					ith.pres = ith.EoS->Pressure(ith.dens, 0.0);
					ith.snds = ith.EoS->SoundSpeed(ith.dens, 0.0);
					ith.mass = ith.dens * dx * dx * dx;
					ith.initialize();
					sph_system.setNumberOfParticleLocal(sph_system.getNumberOfParticleLocal() + 1);
					sph_system[sph_system.getNumberOfParticleLocal() - 1] = ith;
					mass += ith.mass;
				}
			}
			std::cout << "inject rate " << mass / total_mass << std::endl;
			time += dx / velocity.x;
		}
	}
};

template <class Ptcl> class Dam2: public Problem<Ptcl>{
	public:
	static constexpr double END_TIME = 6.5;
	static void setupIC(PS::ParticleSystem<Ptcl>& sph_system, system_t& sysinfo, PS::DomainInfo& dinfo){
		/////////
		//place ptcls
		/////////
		system_t dummy;
		char filename[256];
		sprintf(filename, "init/init_%05d_%05d.bin", PS::Comm::getNumberOfProc(), PS::Comm::getRank());
		std::ifstream fin(filename, std::ios::in | std::ios::binary);
		if(!fin){
			std::cout << "cannot open restart file." << std::endl;
			exit(1);
		}
		std::cout << "open" << std::endl;
		fin.read((char*)&dummy, sizeof(system_t));
		std::vector<Ptcl> ptcl;
		while(1){
			Ptcl ith;
			fin.read((char*)&ith, sizeof(Ptcl));
			if(fin.eof() == true) break;
			ith.EoS = &EoS::Water;
			ith.visc = &Viscosity::Water;
			ptcl.push_back(ith);
		}
		fin.close();
		sph_system.setNumberOfParticleLocal(ptcl.size());
		for(std::size_t i = 0 ; i < ptcl.size() ; ++ i){
			sph_system[i] = ptcl[i];
		}
	}
	static void addExternalForce(PS::ParticleSystem<Ptcl>& sph_system, system_t& system){
		for(PS::U64 i = 0 ; i < sph_system.getNumberOfParticleLocal() ; ++ i){
			sph_system[i].acc.z -= 9.8;
			if(system.time > 3.0) continue;
			if(system.dt != 0.0) sph_system[i].acc += - 0.1 * sph_system[i].vel / system.dt;
		}
	}
	static void postTimestepProcess(PS::ParticleSystem<Ptcl>& sph_system, system_t& system){
		for(PS::U64 i = 0 ; i < sph_system.getNumberOfParticleLocal() ; ++ i){
			sph_system[i].dens = std::max(sph_system[i].dens, 100.);
		}
	}
};

