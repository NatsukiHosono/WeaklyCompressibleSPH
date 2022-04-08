#include <particle_simulator.hpp>
#include <sys/time.h>
#include <chrono>
//#if __CUDA_ARCH__ >= 700
//#warning ARCH 7.0
#include "half.h"
//#endif
#include "param.h"
#include "EoS.h"
#include "kernel.h"
#include "class.h"
#include "kernel.cuh"
#include "prototype.h"

struct CudaTimer{
	float wc_time;
	cudaEvent_t m_start_event;
	cudaEvent_t m_stop_event;
	CudaTimer(){
		wc_time = 0.0;
		cudaEventCreate(&m_start_event);
		cudaEventCreate(&m_stop_event);
	}
	~CudaTimer(){
		cudaEventDestroy(m_start_event);
		cudaEventDestroy(m_stop_event);
	}
	void start(){
		cudaEventRecord(m_start_event, 0);
	}
	void stop(){
		cudaEventRecord(m_stop_event, 0);
		cudaEventSynchronize(m_stop_event);
		//milliseconds
		cudaEventElapsedTime(&wc_time, m_start_event, m_stop_event);
	}
	float getWallclockTime() const{
		return wc_time / 1.0e+3;
	}
};

namespace GPU_WC{
	namespace Hydr{
		template <typename real> struct EpiDev{
			real rx;
			real ry;
			real rz;
			real vx;
			real vy;
			real vz;
			real dens;
			real pres;
			real snds;
			real visc;
			real smth;
			int  id_walk;
			template <typename real_rhs> inline __device__ EpiDev<real>(const EpiDev<real_rhs>& rhs){
				rx = rhs.rx;
				ry = rhs.ry;
				rz = rhs.rz;
				vx = rhs.vx;
				vy = rhs.vy;
				vz = rhs.vz;
				dens = rhs.dens;
				pres = rhs.pres;
				snds = rhs.snds;
				visc = rhs.visc;
				smth = rhs.smth;
				id_walk = rhs.id_walk;
			}
		};

		template <typename real> struct EpjDev{
			real rx;
			real ry;
			real rz;
			real vx;
			real vy;
			real vz;
			real dens;
			real pres;
			real snds;
			real visc;
			real mass;
			real smth;
			template <typename real_rhs> inline __device__ EpjDev<real>(const EpjDev<real_rhs>& rhs){
				rx = rhs.rx;
				ry = rhs.ry;
				rz = rhs.rz;
				vx = rhs.vx;
				vy = rhs.vy;
				vz = rhs.vz;
				dens = rhs.dens;
				pres = rhs.pres;
				snds = rhs.snds;
				visc = rhs.visc;
				mass = rhs.mass;
				smth = rhs.smth;
			}
		};

		template <typename real> struct ForceDev{
			real ax;
			real ay;
			real az;
			real div_v;
			real dt;
		};
	}
	template <typename real, typename real_force = real> struct ptr_t{
		int *ni_displc, *nj_displc;
		Hydr::EpiDev<real> *epi;
		Hydr::EpjDev<real> *epj;
		Hydr::ForceDev<real_force> *res;
		int allocateOnHost(const int N_walkmax, const int Ni_max, const int Nj_max){
			cudaMallocHost((void**)&ni_displc, (N_walkmax + 1) * sizeof(int));
			cudaMallocHost((void**)&nj_displc, (N_walkmax + 1) * sizeof(int));
			cudaMallocHost((void**)&epi      , Ni_max * sizeof(Hydr::EpiDev<real>));
			cudaMallocHost((void**)&epj      , Nj_max * sizeof(Hydr::EpjDev<real>));
			cudaMallocHost((void**)&res      , Ni_max * sizeof(Hydr::ForceDev<real_force>));
			return 0;
		}
		int allocateOnDevice(const int N_walkmax, const int Ni_max, const int Nj_max){
			cudaMalloc((void**)&ni_displc, (N_walkmax + 1) * sizeof(int));
			cudaMalloc((void**)&nj_displc, (N_walkmax + 1) * sizeof(int));
			cudaMalloc((void**)&epi      , Ni_max * sizeof(Hydr::EpiDev<real>));
			cudaMalloc((void**)&epj      , Nj_max * sizeof(Hydr::EpjDev<real>));
			cudaMalloc((void**)&res      , Ni_max * sizeof(Hydr::ForceDev<real_force>));
			return 0;
		}
	};

	template <typename fp> struct /* __device_builtin__ */ __builtin_align__(sizeof(fp) * 4) fp4{
		fp x, y, z, w;
		__device__ fp4(const fp x_, const fp y_, const fp z_, const fp w_): x(x_), y(y_), z(z_), w(w_){
		}
		inline __device__ fp4<fp> operator/(const fp& b) const{
			return fp4<fp>(x / b, y / b, z / b, w / b);
		}
		inline __device__ fp operator*(const fp4<fp>& b) const{
			return x * b.x + y * b.y + z * b.z + w * b.w;
		}
		inline __device__ fp4<fp> operator+(const fp4<fp>& b) const{
			return fp4<fp>(x + b.x, y + b.y, z + b.z, w + b.w);
		}
		template <typename fp_l> inline __device__ operator fp4<fp_l> () const{
			return fp4<fp_l>(x, y, z, w);
		}
	};
	template <typename fp> inline __device__ fp4<fp> operator*(const fp& b, const fp4<fp>& a){
		return fp4<fp>(b * a.x, b * a.y, b * a.z, b * a.w);
	}
	/*
	template <typename real> struct CubicSpline{
		__host__ __device__ static real plus(const real x){
			return (x > 0) ? x : static_cast<real>(0);
		}
		__host__ __device__ static real pow2(const real arg){
			return arg * arg;
		}
		__host__ __device__ static real normalise(const real h){
			const real H = supportRadius() * h;
			#ifdef PARTICLE_SIMULATOR_TWO_DIMENSION
			return static_cast<real>(static_cast<real>(80.) / (static_cast<real>(7.) * H * H) * static_cast<real>(M_1_PI));
			#else
			return static_cast<real>(static_cast<real>(16.) / (H * H * H) * static_cast<real>(M_1_PI));
			#endif
		}
		//gradW
		template <typename real2> __host__ __device__ static fp4<real> gradW(const fp4<real2> dr, const real h){
			const real H = supportRadius() * h;
			const real r = sqrt(dr.x * dr.x + dr.y * dr.y + dr.z * dr.z);
			const real s = r / H;
			real r_value = - static_cast<real>(3.0) * pow2(plus(static_cast<real>(1.0) - s)) + static_cast<real>(12.0) * pow2(plus(static_cast<real>(0.5) - s));
			return r_value * fp4<real>(dr.x, dr.y, dr.z, dr.w) / (r * H);
		}
		__host__ __device__ static real supportRadius(){
			return static_cast<real>(2.0);
		}
	};
	*/
	template <typename real> struct CubicSpline{
		__host__ __device__ static real plus(const real x){
			return (x > 0) ? x : static_cast<real>(0);
		}
		__host__ __device__ static real pow2(const real arg){
			return arg * arg;
		}
		__host__ __device__ static real normalise(const real h){
			const real H = supportRadius() * h;
			#ifdef PARTICLE_SIMULATOR_TWO_DIMENSION
			return static_cast<real>(static_cast<real>(80.) / (static_cast<real>(7.) * H * H * H) * static_cast<real>(M_1_PI));
			#else
			return static_cast<real>(static_cast<real>(16.) / (H * H * H * H) * static_cast<real>(M_1_PI));
			#endif
		}
		//gradW
		template <typename real2> __host__ __device__ static fp4<real> gradW(const fp4<real2> dr, const real h){
			const real H = supportRadius() * h;
			const real r = sqrt(dr.x * dr.x + dr.y * dr.y + dr.z * dr.z);
			const real s = r / H;
			real r_value = - static_cast<real>(3.0) * pow2(plus(static_cast<real>(1.0) - s)) + static_cast<real>(12.0) * pow2(plus(static_cast<real>(0.5) - s));
			return r_value * fp4<real>(dr.x, dr.y, dr.z, dr.w) / r;
		}
		__host__ __device__ static real supportRadius(){
			return static_cast<real>(2.0);
		}
	};
	template <typename real> struct WendlandC2{
		__host__ __device__ static real plus(const real x){
			return (x > 0) ? x : static_cast<real>(0);
		}
		__host__ __device__ static real pow3(const real arg){
			return arg * arg * arg;
		}
		__host__ __device__ static real normalise(const real h){
			const real H = supportRadius() * h;
			#ifdef PARTICLE_SIMULATOR_TWO_DIMENSION
			return static_cast<real>(7.0) * static_cast<real>(M_1_PI) / (H * H);
			#else
			return static_cast<real>(10.5) * static_cast<real>(M_1_PI) / (H * H * H);
			#endif
		}
		//gradW
		template <typename real2> __host__ __device__ static fp4<real> gradW(const fp4<real2> dr, const real h){
			const real H = supportRadius() * h;
			const real r = sqrt(dr.x * dr.x + dr.y * dr.y + dr.z * dr.z);
			const real s = r / H;
			real r_value = pow3(plus(static_cast<real>(1.0) - s)) * (-static_cast<real>(4.0) * (static_cast<real>(1.0) + static_cast<real>(4.0) * s) + static_cast<real>(4.0) * plus(static_cast<real>(1.0) - s));
			return r_value * fp4<real>(dr.x, dr.y, dr.z, dr.w) / (r * H);
		}
		__host__ __device__ static real supportRadius(){
			return static_cast<real>(2.0);
		}
	};

	typedef half fp;
	typedef half fp_force;
	ptr_t<fp, fp_force> host, device;
	double smth_glb, mass_glb;
	template <typename real, class kernel_t, typename real_force = real> __global__ void deviceCalcHydrForce(const Hydr::EpiDev<real> *epi, const int *ni_displc, const Hydr::EpjDev<real> *epj, const int *nj_displc, Hydr::ForceDev<real_force> *force){
		const int id = blockDim.x * blockIdx.x + threadIdx.x;
		const Hydr::EpiDev<real_force> ith = epi[id];
		const int j_head = nj_displc[ith.id_walk];
		const int j_tail = nj_displc[ith.id_walk + 1];

		fp4<real_force> force_buf(0.0, 0.0, 0.0, 0.0);
		//const real_force ith_pres_over_dens2 = ith.pres / (ith.dens * ith.dens);
		const real_force AV_STRENGTH = 0.01;
		real_force v_sig_max = 0.0;
		for(int j = j_head ; j < j_tail ; ++ j){
			const Hydr::EpjDev<real_force> jth = epj[j];
			const fp4<real_force> dr(jth.rx - ith.rx, jth.ry - ith.ry, jth.rz - ith.rz, 0.0);
			const fp4<real_force> dv(jth.vx - ith.vx, jth.vy - ith.vy, jth.vz - ith.vz, 0.0);
			const real_force r = sqrt(dr * dr);
			if(r <= static_cast<real_force>(0.0)) continue;

			const real_force drdv = dr * dv;
			const real_force c_ij = static_cast<real_force>(0.5) * (ith.snds + jth.snds);
			const real_force dens_ij = static_cast<real_force>(0.5) * (ith.dens + jth.dens);
			const real_force mu_ij = (drdv < 0) ? static_cast<real_force>(0.5) * (ith.smth + jth.smth) * drdv / (r * r + static_cast<real_force>(0.01) * ith.smth * jth.smth) : static_cast<real_force>(0.0);
			const real_force AV = (- AV_STRENGTH * c_ij * mu_ij + AV_STRENGTH * static_cast<real_force>(2.0) * mu_ij * mu_ij) / dens_ij;
			v_sig_max = max(v_sig_max, - mu_ij);
			
			const fp4<real_force> gradW = static_cast<real_force>(0.5) * (kernel_t::gradW(dr, ith.smth) + kernel_t::gradW(dr, jth.smth));
			//const real_force acc = (ith_pres_over_dens2 + jth.pres / (jth.dens * jth.dens) + AV);
			const real_force acc = (ith.pres + jth.pres + AV);
			const real_force visc = static_cast<real_force>(2.0) * (ith.visc + jth.visc) * dr * gradW / (ith.dens + jth.dens) / (r * r + static_cast<real_force>(0.01) * ith.smth * jth.smth);

			force_buf.x += acc * gradW.x - visc * dv.x;
			force_buf.y += acc * gradW.y - visc * dv.y;
			force_buf.z += acc * gradW.z - visc * dv.z;
			force_buf.w += dv * gradW;
		}
		force[id].ax = force_buf.x;
		force[id].ay = force_buf.y;
		force[id].az = force_buf.z;
		force[id].div_v = - force_buf.w / ith.dens;
		force[id].dt = static_cast<real_force>(PARAM::C_CFL) * ith.smth / (ith.snds + v_sig_max);
	}
	//const double v0 = 0.00625;//typical velocity
	//const double rho0 = 1000.0;//typical density
	//const double L = 0.2;//typical length
	const double v0 = sqrt(2.0 * 9.8 * 0.55);//typical velocity
	const double rho0 = 1000.0;//typical density
	const double L = 0.55;//typical length
	const double p0 = rho0 * v0 * v0;//typical pres
	const double visc0 = v0 * L;
	PS::S32 HydrDispatchKernel(const PS::S32 tag, const int n_walk, const STD::EPI::Hydro** epi, const int* n_epi, const STD::EPJ::Hydro** epj, const int* n_epj){
		static bool isFirst = true;
		if(isFirst == true){
			#ifdef PARTICLE_SIMULATOR_MPI_PARALLEL
			cudaSetDevice(PS::Comm::getRank());
			#else
			std::cout << "input CUDA device: " << std::endl;
			int deviceId = 0;
			std::cin >> deviceId;
			cudaSetDevice(deviceId);
			#endif
			device.allocateOnDevice(N_WALK_LIMIT, NI_LIMIT, NJ_LIMIT);
			host.allocateOnHost(N_WALK_LIMIT, NI_LIMIT, NJ_LIMIT);
			isFirst = false;
			smth_glb = epi[0][0].smth;
			mass_glb = epj[0][0].mass;
		}
		host.ni_displc[0] = host.nj_displc[0] = 0;
		for(std::size_t i = 0; i < n_walk ; ++ i){
			host.ni_displc[i+1] = host.ni_displc[i] + n_epi[i];
			host.nj_displc[i+1] = host.nj_displc[i] + n_epj[i];
		}
		const PS::S32 ni_total = host.ni_displc[n_walk];
		const int ni_total_reg = host.ni_displc[n_walk] + ((ni_total % N_THREAD_GPU != 0) ? (N_THREAD_GPU - (ni_total % N_THREAD_GPU)) : 0);
		//make data for device on host
		int cnt = 0;
		int cnt_j = 0;
		assert(ni_total < NI_LIMIT);
		for(std::size_t walk = 0 ; walk < n_walk ; ++ walk){
			PS::F64vec com = 0.0;
			PS::F64vec vel = 0.0;
			PS::F64 mass = 0.0;
			for(std::size_t j = 0 ; j < n_epj[walk] ; ++ j){
				com += epj[walk][j].pos * epj[walk][j].mass;
				vel += epj[walk][j].vel * epj[walk][j].mass;
				mass += epj[walk][j].mass;
			}
			com /= mass;
			vel /= mass;
			for(std::size_t i = 0 ; i < n_epi[walk] ; ++ i){
				host.epi[cnt].rx      = (epi[walk][i].pos.x - com.x) / L;
				host.epi[cnt].ry      = (epi[walk][i].pos.y - com.y) / L;
				#ifndef PARTICLE_SIMULATOR_TWO_DIMENSION
				host.epi[cnt].rz      = (epi[walk][i].pos.z - com.z) / L;
				#endif
				host.epi[cnt].vx      = (epi[walk][i].vel.x - vel.x) / v0;
				host.epi[cnt].vy      = (epi[walk][i].vel.y - vel.y) / v0;
				#ifndef PARTICLE_SIMULATOR_TWO_DIMENSION
				host.epi[cnt].vz      = (epi[walk][i].vel.z - vel.z) / v0;
				#endif
				host.epi[cnt].dens    = epi[walk][i].dens / rho0;
				//host.epi[cnt].pres    = epi[walk][i].pres / p0;
				host.epi[cnt].pres    = epi[walk][i].pres / p0 / pow(epi[walk][i].dens / rho0, 2);
				host.epi[cnt].snds    = epi[walk][i].snds / v0;
				host.epi[cnt].visc    = epi[walk][i].visc / visc0;
				host.epi[cnt].smth    = epi[walk][i].smth / L;
				host.epi[cnt].id_walk = walk;
				++ cnt;
			}
			for(std::size_t j = 0 ; j < n_epj[walk] ; ++ j){
				host.epj[cnt_j].rx   = (epj[walk][j].pos.x - com.x) / L;
				host.epj[cnt_j].ry   = (epj[walk][j].pos.y - com.y) / L;
				#ifndef PARTICLE_SIMULATOR_TWO_DIMENSION
				host.epj[cnt_j].rz   = (epj[walk][j].pos.z - com.z) / L;
				#endif
				host.epj[cnt_j].vx   = (epj[walk][j].vel.x - vel.x) / v0;
				host.epj[cnt_j].vy   = (epj[walk][j].vel.y - vel.y) / v0;
				#ifndef PARTICLE_SIMULATOR_TWO_DIMENSION
				host.epj[cnt_j].vz   = (epj[walk][j].vel.z - vel.z) / v0;
				#endif
				host.epj[cnt_j].dens = epj[walk][j].dens / rho0;
				//host.epj[cnt_j].pres = epj[walk][j].pres / p0;
				host.epj[cnt_j].pres = epj[walk][j].pres / p0 / pow(epj[walk][j].dens / rho0, 2);
				host.epj[cnt_j].snds = epj[walk][j].snds / v0;
				host.epj[cnt_j].visc = epj[walk][j].visc / visc0;
				host.epj[cnt_j].mass = epj[walk][j].mass / (rho0 * L * L * L);
				host.epj[cnt_j].smth = epj[walk][j].smth / L;
				++ cnt_j;
				assert(cnt_j < NJ_LIMIT);
			}
		}
		cudaMemcpy(device.epi      , host.epi      , ni_total_reg * sizeof(Hydr::EpiDev<fp>), cudaMemcpyHostToDevice);
		cudaMemcpy(device.epj      , host.epj      , cnt_j * sizeof(Hydr::EpjDev<fp>), cudaMemcpyHostToDevice);
		cudaMemcpy(device.ni_displc, host.ni_displc, (n_walk + 1) * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(device.nj_displc, host.nj_displc, (n_walk + 1) * sizeof(int), cudaMemcpyHostToDevice);

		const int n_grid = ni_total_reg / N_THREAD_GPU + ((ni_total_reg % N_THREAD_GPU == 0) ? 0 : 1);
		dim3 size_grid(n_grid, 1, 1);
		dim3 size_thread(N_THREAD_GPU, 1, 1);
		deviceCalcHydrForce<fp, CubicSpline<fp_force>, fp_force> <<<size_grid, size_thread>>> (device.epi, device.ni_displc, device.epj, device.nj_displc, device.res);
		//deviceCalcHydrForce<fp, WendlandC2<fp_force>, fp_force> <<<size_grid, size_thread>>> (device.epi, device.ni_displc, device.epj, device.nj_displc, device.res);
		return 0;
	}

	PS::S32 HydrRetrieveKernel(const PS::S32 tag, const PS::S32 n_walk, const PS::S32* ni, STD::RESULT::Hydro** force){
		int ni_tot = 0;
		for(int i = 0 ; i < n_walk ; ++ i){
			ni_tot += ni[i];
		}
		cudaMemcpy(host.res, device.res, ni_tot * sizeof(Hydr::ForceDev<fp_force>), cudaMemcpyDeviceToHost);
		int cnt = 0;
		for(int walk = 0 ; walk < n_walk ; ++ walk){
			for(int i = 0 ; i < ni[walk] ; ++ i){
				force[walk][i].acc.x = host.res[cnt].ax;
				force[walk][i].acc.y = host.res[cnt].ay;
				#ifndef PARTICLE_SIMULATOR_TWO_DIMENSION
				force[walk][i].acc.z = host.res[cnt].az;
				#endif
				force[walk][i].div_v = host.res[cnt].div_v;
				force[walk][i].acc *= v0 * v0 / rho0;
				force[walk][i].div_v *= v0 / rho0;
				force[walk][i].acc *= CubicSpline<double>::normalise(smth_glb) * mass_glb;
				force[walk][i].div_v *= CubicSpline<double>::normalise(smth_glb) * mass_glb;
				//force[walk][i].acc *= WendlandC2<double>::normalise(smth_glb) * mass_glb;
				//force[walk][i].div_v *= WendlandC2<double>::normalise(smth_glb) * mass_glb;
				force[walk][i].dt = static_cast<double>(host.res[cnt].dt) * L / v0;
				++ cnt;
			}
		}
		return 0;
	}

};

