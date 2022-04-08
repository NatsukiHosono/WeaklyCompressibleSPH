#pragma once

class KernelFunction{
	protected:
	static constexpr PS::F64 pi = M_PI;
	static PS::F64 plus(const PS::F64 arg){
		return (arg > 0) ? arg : 0;
	}
	static PS::F64 pow2(const PS::F64 arg){
		return arg * arg;
	}
	static PS::F64 pow3(const PS::F64 arg){
		const PS::F64 arg3 = arg * arg * arg;
		return arg3;
	}
	static PS::F64 pow4(const PS::F64 arg){
		const PS::F64 arg2 = arg * arg;
		return arg2 * arg2;
	}
	static PS::F64 normalisedConstant();
	public:
	static PS::F64 W(const PS::F64vec dr, const PS::F64 h);
	static PS::F64vec gradW(const PS::F64vec dr, const PS::F64 h);
	static PS::F64 supportRadius();
};

template <int Ndim> class CubicSpline : public KernelFunction{
	public:
	static PS::F64 supportRadius(){
		return 2.0;
	}
	static PS::F64 normalisedConstant(){
		if(Ndim == 2){
			return 80. / (7. * pi);
		}else{
			return 16. / pi;
		}
	}
	static PS::F64 W(const PS::F64vec dr, const PS::F64 h){
		const PS::F64 H = supportRadius() * h;
		const PS::F64 s = sqrt(dr * dr) / H;
		PS::F64 r_value;
		r_value = pow3(plus(1.0 - s)) - 4.0 * pow3(plus(0.5 - s));
		r_value *= normalisedConstant() / pow(H, Ndim);
		return r_value;
	}
	static PS::F64vec gradW(const PS::F64vec dr, const PS::F64 h){
		const PS::F64 H = supportRadius() * h;
		const PS::F64 s = sqrt(dr * dr) / H;
		PS::F64 r_value;
		r_value = - 3.0 * pow2(plus(1.0 - s)) + 12.0 * pow2(plus(0.5 - s));
		r_value *= normalisedConstant() / pow(H, Ndim);
		return dr * r_value / (sqrt(dr * dr) * H);
	}
};

template <int Ndim> class WendlandC2 : public KernelFunction{
	public:
	static PS::F64 supportRadius(){
		return 2.0;
	}
	static PS::F64 normalisedConstant(){
		if(Ndim == 2){
			return 7.0 / pi;
		}else{
			return 10.5 / pi;
		}
	}
	static PS::F64 W(const PS::F64vec dr, const PS::F64 h){
		const PS::F64 H = supportRadius() * h;
		const PS::F64 s = sqrt(dr * dr) / H;
		PS::F64 r_value;
		r_value = (1.0 + 4.0 * s) * pow4(plus(1.0 - s));
		r_value *= normalisedConstant() / pow(H, Ndim);
		return r_value;
	}
	static PS::F64vec gradW(const PS::F64vec dr, const PS::F64 h){
		const PS::F64 H = supportRadius() * h;
		const PS::F64 s = sqrt(dr * dr) / H;
		PS::F64 r_value;
		r_value = pow3(plus(1.0 - s)) * (-4.0 * (1.0 + 4.0 * s) + 4.0 * plus(1.0 - s));
		r_value *= normalisedConstant() / pow(H, Ndim);
		return dr * r_value / (sqrt(dr * dr) * H);
	}
};

//Wendland C6
struct WendlandC6{
	static constexpr PS::F64 pi = M_PI;
	PS::F64 plus(const PS::F64 arg) const{
		return (arg > 0) ? arg : 0;
	}
	PS::F64 pow7(const PS::F64 arg) const{
		const PS::F64 arg3 = arg * arg * arg;
		return arg3 * arg3 * arg;
	}
	PS::F64 pow8(const PS::F64 arg) const{
		const PS::F64 arg2 = arg * arg;
		const PS::F64 arg4 = arg2 * arg2;
		return arg4 * arg4;
	}
	WendlandC6(){}
	//W
	PS::F64 W(const PS::F64vec dr, const PS::F64 h) const{
		const PS::F64 H = supportRadius() * h;
		const PS::F64 s = sqrt(dr * dr) / H;
		PS::F64 r_value;
		r_value = (1.0 + s * (8.0 + s * (25.0 + s * (32.0)))) * pow8(plus(1.0 - s));
		#ifdef PARTICLE_SIMULATOR_TWO_DIMENSION
		r_value *= (78./7.) / (H * H * pi);
		#else
		r_value *= (1365./64.) / (H * H * H * pi);
		#endif
		return r_value;
	}
	//gradW
	PS::F64vec gradW(const PS::F64vec dr, const PS::F64 h) const{
		const PS::F64 H = supportRadius() * h;
		const PS::F64 s = sqrt(dr * dr) / H;
		PS::F64 r_value;
		r_value = pow7(plus(1.0 - s)) * (plus(1.0 - s) * (8.0 + s * (50.0 + s * (96.0))) - 8.0 * (1.0 + s * (8.0 + s * (25.0 + s * (32.0)))));
		#ifdef PARTICLE_SIMULATOR_TWO_DIMENSION
		r_value *= (78./7.) / (H * H * pi);
		#else
		r_value *= (1365./64.) / (H * H * H * pi);
		#endif
		return dr * r_value / (sqrt(dr * dr) * H  + 1.0e-6 * h);
	}
	static PS::F64 supportRadius(){
		return 2.0;
	}
};

struct WendlandC4{
	static constexpr PS::F64 pi = M_PI;
	PS::F64 plus(const PS::F64 arg) const{
		return (arg > 0) ? arg : 0;
	}
	PS::F64 pow5(const PS::F64 arg) const{
		const PS::F64 arg2 = arg * arg;
		return arg2 * arg2 * arg;
	}
	PS::F64 pow6(const PS::F64 arg) const{
		const PS::F64 arg2 = arg * arg;
		return arg2 * arg2 * arg2;
	}
	WendlandC4(){}
	//W
	PS::F64 W(const PS::F64vec dr, const PS::F64 h) const{
		const PS::F64 H = supportRadius() * h;
		const PS::F64 s = sqrt(dr * dr) / H;
		PS::F64 r_value;
		r_value = (1.0 + 6.0 * s + 35.0 / 3.0 * s * s) * pow6(plus(1.0 - s));
		#ifdef PARTICLE_SIMULATOR_TWO_DIMENSION
		r_value *= 9. / (H * H * pi);
		#else
		r_value *= 495./32. / (H * H * H * pi);
		#endif
		return r_value;
	}
	//gradW
	PS::F64vec gradW(const PS::F64vec dr, const PS::F64 h) const{
		const PS::F64 H = supportRadius() * h;
		const PS::F64 s = sqrt(dr * dr) / H;
		PS::F64 r_value;
		r_value = pow5(plus(1.0 - s)) * (-6.0 * (1.0 + 6.0 * s + 35.0 / 3.0 * s * s) + (6.0 + 70.0 / 3.0 * s) * plus(1.0 - s));
		#ifdef PARTICLE_SIMULATOR_TWO_DIMENSION
		r_value *= 9. / (H * H * pi);
		#else
		r_value *= 495./32. / (H * H * H * pi);
		#endif
		return dr * r_value / (sqrt(dr * dr) * H);
	}
	static PS::F64 supportRadius(){
		return 2.0;
	}
};


struct Quartic{
	static constexpr PS::F64 pi = M_PI;
	PS::F64 plus(const PS::F64 arg) const{
		return (arg > 0) ? arg : 0;
	}
	PS::F64 pow3(const PS::F64 arg) const{
		const PS::F64 arg3 = arg * arg * arg;
		return arg3;
	}
	PS::F64 pow4(const PS::F64 arg) const{
		const PS::F64 arg2 = arg * arg;
		return arg2 * arg2;
	}
	Quartic(){}
	//W
	PS::F64 W(const PS::F64vec dr, const PS::F64 h) const{
		const PS::F64 H = supportRadius() * h;
		const PS::F64 s = sqrt(dr * dr) / H;
		PS::F64 r_value;
		r_value = pow4(plus(1.0 - s)) - 5.0 * pow4(plus(3./5. - s)) + 10.0 * pow4(plus(1.0 / 5.0 - s));
		#ifdef PARTICLE_SIMULATOR_TWO_DIMENSION
		r_value *= 46875. / (2398. * H * H * pi);
		#else
		r_value *= 15625. / (512. * H * H * H * pi);
		#endif
		return r_value;
	}
	//gradW
	PS::F64vec gradW(const PS::F64vec dr, const PS::F64 h) const{
		const PS::F64 H = supportRadius() * h;
		const PS::F64 s = sqrt(dr * dr) / H;
		PS::F64 r_value;
		r_value = - 4.0 * pow3(plus(1.0 - s)) + 12.0 * pow3(plus(3./5. - s)) - 8.0 * pow3(plus(1.0 / 5.0 - s));
		#ifdef PARTICLE_SIMULATOR_TWO_DIMENSION
		r_value *= 46875. / (2398. * H * H * pi);
		#else
		r_value *= 15625. / (512. * H * H * H * pi);
		#endif
		return dr * r_value / (sqrt(dr * dr) * H);
	}
	static PS::F64 supportRadius(){
		return 2.0;
	}
};

typedef CubicSpline<PS::DIMENSION> kernel_t;

