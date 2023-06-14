#include <particle_simulator.hpp>
#include <sys/time.h>
#include <typeinfo>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>

#ifdef ENABLE_GPU
#include "kernel.cuh"
#endif

#include "param.h"
#include "kernel.h"
#include "EoS.h"
#include "class.h"
#include "init/Dam2.h"
#include "force.h"
#include "io.h"
#include "integral.h"
#include "prototype.h"

int main(int argc, char* argv[]){
	namespace PTCL = STD;
	namespace MODE = GPU_WC;
	typedef Dam<PTCL::RealPtcl> PROBLEM;
	//////////////////
	//Create vars.
	//////////////////
	PS::Initialize(argc, argv);
	PS::ParticleSystem<PTCL::RealPtcl> sph_system;
	sph_system.initialize();
	PS::DomainInfo dinfo;
	dinfo.initialize(0.3);
	system_t sysinfo;
	sph_system.setAverageTargetNumberOfSampleParticlePerProcess(200);

	//////////////////
	//Setup Initial
	//////////////////
	if(argc == 1){
		PROBLEM::setupIC(sph_system, sysinfo, dinfo);
	}else{
		sysinfo.step = atoi(argv[1]);
		InputBinary<PTCL::RealPtcl>(sph_system, &sysinfo);
		PS::Comm::barrier();
	}
	#pragma omp parallel for
	for(PS::S32 i = 0 ; i < sph_system.getNumberOfParticleLocal() ; ++ i){
		sph_system[i].initialize();
	}
	//Display info
	if(PS::Comm::getRank() == 0){
		std::cout << PS::Comm::getNumberOfThread() << std::endl;
	}

	//Domain info
	dinfo.decomposeDomainAll(sph_system);
	sph_system.exchangeParticle(dinfo);
	//plant tree
	PS::TreeForForceShort<PTCL::RESULT::Hydro, PTCL::EPI::Hydro, PTCL::EPJ::Hydro>::Symmetry hydr_tree;
	PS::TreeForForceShort<PTCL::RESULT::Filter, PTCL::EPI::Filter, PTCL::EPJ::Filter>::Gather filt_tree;

	hydr_tree.initialize(sph_system.getNumberOfParticleLocal(), 0.5, PARAM::Nleaf, PARAM::Ngroup);
	filt_tree.initialize(sph_system.getNumberOfParticleLocal(), 0.5, PARAM::Nleaf, PARAM::Ngroup);
	PTCL::CalcPressure(sph_system);
	#ifdef ENABLE_GPU
		hydr_tree.calcForceAllAndWriteBackMultiWalk(MODE::HydrDispatchKernel, MODE::HydrRetrieveKernel, 1, sph_system, dinfo, N_WALK_LIMIT, true, PS::MAKE_LIST_FOR_REUSE);
	#else
		hydr_tree.calcForceAllAndWriteBack(PTCL::CalcHydroForce<kernel_t>(), sph_system, dinfo, PS::MAKE_LIST_FOR_REUSE);
	#endif
	filt_tree.calcForceAllAndWriteBack(PTCL::ApplyFilter<kernel_t>(), sph_system, dinfo);

	PROBLEM::addExternalForce(sph_system, sysinfo);
	PROBLEM::postTimestepProcess(sph_system, sysinfo);
	OutputFileWithTimeInterval(sph_system, sysinfo, PROBLEM::END_TIME);

	if(PS::Comm::getRank() == 0){
		std::cout << "//================================" << std::endl;
		std::cout << std::scientific << std::setprecision(16) << "time = " << sysinfo.time << ", dt = " << sysinfo.dt << std::endl;
		std::cout << "step = " << sysinfo.step << std::endl;
		std::cout << "//================================" << std::endl;
	}

	Timer timer;
	while(sysinfo.time < PROBLEM::END_TIME){
		sysinfo.dt = getTimeStepGlobal<PTCL::RealPtcl>(sph_system);
		if(PS::Comm::getRank() == 0 && sysinfo.step % 1 == 0){
			std::cout << std::scientific << std::setprecision(16) << "time = " << sysinfo.time << ", dt = " << sysinfo.dt << std::endl;
			std::cout << "step = " << sysinfo.step << std::endl;
			timer.stop();
			timer.dumpWCTime();
			timer.start();
		}
		#pragma omp parallel for
		for(int i = 0 ; i < sph_system.getNumberOfParticleLocal() ; ++ i){
			sph_system[i].initialKick(sysinfo.dt);
			sph_system[i].fullDrift(sysinfo.dt);
			sph_system[i].predict(sysinfo.dt);
		}
		sysinfo.time += sysinfo.dt;
		if(sysinfo.step % PARAM::FILTER_APPLYING_INTERVAL == 0) filt_tree.calcForceAllAndWriteBack(PTCL::ApplyFilter<kernel_t>(), sph_system, dinfo);
		PTCL::CalcPressure(sph_system);
		sph_system.adjustPositionIntoRootDomain(dinfo);
		if(sysinfo.step % PARAM::REUSE_LIST_INTERVAL == 0){
			dinfo.decomposeDomainAll(sph_system);
			sph_system.exchangeParticle(dinfo);
			#ifdef ENABLE_GPU
			hydr_tree.calcForceAllAndWriteBackMultiWalk(MODE::HydrDispatchKernel, MODE::HydrRetrieveKernel, 1, sph_system, dinfo, N_WALK_LIMIT, true, PS::MAKE_LIST_FOR_REUSE);
			#else
			hydr_tree.calcForceAllAndWriteBack(PTCL::CalcHydroForce<kernel_t>(), sph_system, dinfo, PS::MAKE_LIST_FOR_REUSE);
			#endif
		}else{
			#ifdef ENABLE_GPU
			hydr_tree.calcForceAllAndWriteBackMultiWalk(MODE::HydrDispatchKernel, MODE::HydrRetrieveKernel, 1, sph_system, dinfo, N_WALK_LIMIT, true, PS::REUSE_LIST);
			#else
			hydr_tree.calcForceAllAndWriteBack(PTCL::CalcHydroForce<kernel_t>(), sph_system, dinfo, PS::REUSE_LIST);
			#endif
		}
		PROBLEM::addExternalForce(sph_system, sysinfo);
		#pragma omp parallel for
		for(int i = 0 ; i < sph_system.getNumberOfParticleLocal() ; ++ i){
			sph_system[i].finalKick(sysinfo.dt);
		}

		PROBLEM::postTimestepProcess(sph_system, sysinfo);
		OutputFileWithTimeInterval<PTCL::RealPtcl>(sph_system, sysinfo, PROBLEM::END_TIME);
		//DebugOutputFile<PTCL::RealPtcl>(sph_system, sysinfo);
		++ sysinfo.step;
	}

	PS::Finalize();
	return 0;
}

