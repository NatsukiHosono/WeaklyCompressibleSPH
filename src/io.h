#pragma once
template <class ThisPtcl> void OutputFileWithTimeInterval(PS::ParticleSystem<ThisPtcl>& sph_system, const system_t& sysinfo, const PS::F64 end_time){
	static PS::F64 time = sysinfo.time;
	static PS::S64 step = sysinfo.step;

	if(sysinfo.time >= time){
		char filename[256];
		//Ascii
		FileHeader header;
		header.time = sysinfo.time;
		header.Nbody = sph_system.getNumberOfParticleLocal();
		sprintf(filename, "result/%05d", step);
		sph_system.writeParticleAscii(filename, "%s_%05d_%05d.dat", header);
		if(PS::Comm::getRank() == 0){
			std::cout << "//================================" << std::endl;
			std::cout << "output " << filename << "." << std::endl;
			std::cout << "//================================" << std::endl;
		}
		//Binary
		std::ofstream fout;
		sprintf(filename, "result/%05d_%05d_%05d.bin", step % 10, PS::Comm::getNumberOfProc(), PS::Comm::getRank());
		fout.open(filename, std::ios::out | std::ios::binary | std::ios::trunc);
		if(!fout){
			std::cout << "cannot open restart file." << std::endl;
			exit(1);
		}
		fout.write(reinterpret_cast<const char * const>(&sysinfo), sizeof(system_t));
		for(std::size_t i = 0 ; i < sph_system.getNumberOfParticleLocal() ; ++ i){
			ThisPtcl ith = sph_system[i];
			fout.write((char*)&ith, sizeof(ThisPtcl));
		}
		fout.close();

		time += end_time / PARAM::NUMBER_OF_SNAPSHOTS;
		++ step;
	}
}

template <class ThisPtcl> void InputFileWithTimeInterval(PS::ParticleSystem<ThisPtcl>& sph_system, system_t& sysinfo){
	FileHeader header;
	char filename[256];
	sprintf(filename, "result/%05d", sysinfo.step);
	sph_system.readParticleAscii(filename, "%s_%05d_%05d.dat", header);
	sysinfo.time = header.time;
	std::cout << header.time << std::endl;
}

template <class ThisPtcl> void DebugOutputFile(PS::ParticleSystem<ThisPtcl>& sph_system, const system_t& sysinfo){
	FileHeader header;
	header.time = sysinfo.time;
	header.Nbody = sph_system.getNumberOfParticleLocal();
	char filename[256];
	sprintf(filename, "result/debug%05d", sysinfo.step % 100);
	sph_system.writeParticleAscii(filename, "%s_%05d_%05d.dat", header);
	if(PS::Comm::getRank() == 0){
		std::cout << "DEBUG MODE... " << filename << "." << std::endl;
	}
}

template <class ThisPtcl> void InputBinary(PS::ParticleSystem<ThisPtcl>& sph_system, system_t* sysinfo){
	const int readstep = sysinfo->step;
	char filename[256];
	sprintf(filename, "result/%05d_%05d_%05d.bin", readstep % 100, PS::Comm::getNumberOfProc(), PS::Comm::getRank());
	std::ifstream fin(filename, std::ios::in | std::ios::binary);
	if(!fin){
		std::cout << "cannot open restart file." << std::endl;
		exit(1);
	}
	std::vector<ThisPtcl> ptcl;
	fin.read((char*)sysinfo, sizeof(system_t));
	sysinfo->step = readstep + 1;
	while(1){
		ThisPtcl ith;
		fin.read((char*)&ith, sizeof(ThisPtcl));
		if(fin.eof() == true) break;
		ptcl.push_back(ith);
	}
	fin.close();
	sph_system.setNumberOfParticleLocal(ptcl.size());
	for(std::size_t i = 0 ; i < ptcl.size() ; ++ i){
		sph_system[i] = ptcl[i];
	}
	std::cout << "Input Particle" << std::endl;
}


