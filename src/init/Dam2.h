#pragma once

template <class Ptcl> class Dam: public Problem<Ptcl>{
	public:
	static constexpr double END_TIME = 6.5;
	//material
	/*
	*/
	static void setupIC(PS::ParticleSystem<Ptcl>& sph_system, system_t& sysinfo, PS::DomainInfo& dinfo){
		/////////
		//place ptcls
		/////////
		std::vector<Ptcl> ptcl;
		// set for dam break of http://www.math.rug.nl/~veldman/comflo/dambreak.html
		double NX;
		std::cout << "INPUT NX: " << std::endl;
		std::cin >> NX;
		const PS::F64 dx = 1.0 / NX;//146.0 ;//92.0; //39.0;
		const PS::F64 wall = 0.1;
		const PS::F64 box_x = 3.22;
		const PS::F64 box_y = 1.0;
		const PS::F64 box_z = 1.0;

		//put a sensor
		const PS::F64 offset_x0_sensor = 0.6635;
		const PS::F64 offset_y0_sensor = 0.295;
		const PS::F64 size_x_sensor = 0.161;
		const PS::F64 size_y_sensor = 0.403;
		const PS::F64 size_z_sensor = 0.161;

		//water
		const PS::F64 offset_x0_water = offset_x0_sensor + size_x_sensor + 1.1675;
		const PS::F64 offset_y0_water = 0.0;
		const PS::F64 size_x_water = 1.228;
		const PS::F64 size_y_water = 1.0;
		const PS::F64 size_z_water = 0.55;

		static const double gmm = 7.0;
		static const double rho0 = 1000.0;
		static const double B = 20. * sqrt(9.8 * 2.0 * 0.55) * rho0 / gmm;
		static const double g = 9.8;
		static const EoS::Murnaghan<PS::F64> Water(gmm, rho0, 20. * sqrt(2.0 * g * size_z_water));
		static const EoS::Murnaghan<PS::F64> Wall(gmm, rho0, 20. * sqrt(2.0 * g * size_z_water));

		std::size_t cnt = 0;
		std::size_t cnt_water = 0;
		for(PS::F64 x = - wall ; x <= box_x + wall ; x += dx){
			for(PS::F64 y = - wall ; y <= box_y + wall ; y += dx){
				for(PS::F64 z = - wall ; z <= box_z + wall ; z += dx){
					++ cnt;
					Ptcl ith;
					if(x < 0.0 || y < 0.0 || z < 0.0 || x > box_x || y > box_y){
						//set for boundary
						ith.type = FREEZE;
						ith.tag = 1;
					}else if(offset_x0_sensor <= x && x <= offset_x0_sensor + size_x_sensor && z <= size_z_sensor && offset_y0_sensor <= y && y <= offset_y0_sensor + size_y_sensor){
						ith.type = FREEZE;
						ith.tag = 2;
					}else if(offset_x0_water - wall <= x && x < offset_x0_water && offset_y0_water <= y && y <= offset_y0_water + size_y_water){
						continue;
						ith.type = FREEZE;
						ith.tag = 3;
					}else if(offset_x0_water  <= x && x <= offset_x0_water  + size_x_water  && z <= size_z_water  && offset_y0_water  <= y && y <= offset_y0_water  + size_y_water ){
						ith.type = HYDRO;
						ith.tag = 0;
						cnt_water ++;
					}else{
						continue;
					}
					ith.pos.x = x;
					ith.pos.y = y;
					ith.pos.z = z;
					//ith.dens = EoS::Water.ReferenceDensity() * powf(((gmm - 1) / gmm * g / B * rho0 * std::max(size_z_water - z, 0.0)) + 1.0, 1.0/(gmm - 1.0));
					ith.dens = Water.ReferenceDensity();
					ith.mass = ith.dens * (box_x + 2.0 * wall) * (box_y + 2.0 * wall) * (box_z + 2.0 * wall);
					ith.id = cnt;
					if(ith.type == HYDRO){
						ith.EoS = &Water;
					}else{
						ith.EoS = &Wall;
					}
					ith.visc = &Viscosity::Water;
					ptcl.push_back(ith);
				}
			}
		}

		for(PS::U32 i = 0 ; i < ptcl.size() ; ++ i){
			ptcl[i].mass /= (PS::F64)(cnt);
		}
		std::cout << "# of ptcls is... " << ptcl.size() << " water: " << cnt_water  << std::endl;
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
		}
	}
	static void postTimestepProcess(PS::ParticleSystem<Ptcl>& sph_system, system_t& sysinfo){
		for(PS::U64 i = 0 ; i < sph_system.getNumberOfParticleLocal() ; ++ i){
			sph_system[i].dens = std::max(sph_system[i].dens, 100.);
			//remove wall
			if(sysinfo.time > 0.5 && sph_system[i].tag == 3){
				sph_system[i] = sph_system[sph_system.getNumberOfParticleLocal() - 1];
				sph_system.setNumberOfParticleLocal(sph_system.getNumberOfParticleLocal() - 1);
			}
			//remove runing-away particles
			if(sph_system[i].pos.x < 0 || sph_system[i].pos.y < 0 || sph_system[i].pos.x > 3.5 || sph_system[i].pos.y > 1.0){
				if(sph_system[i].type == FREEZE) continue;
				sph_system[i] = sph_system[sph_system.getNumberOfParticleLocal() - 1];
				sph_system.setNumberOfParticleLocal(sph_system.getNumberOfParticleLocal() - 1);
			}
		}
	}
};

