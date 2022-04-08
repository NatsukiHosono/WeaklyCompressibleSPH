
namespace GPU{
	PS::S32 HydrDispatchKernel(const PS::S32, const PS::S32, const STD::EPI::Hydro**, const PS::S32*, const STD::EPJ::Hydro**, const PS::S32*);
	PS::S32 HydrRetrieveKernel(const PS::S32, const PS::S32, const PS::S32*, STD::RESULT::Hydro**);
};

namespace GPU_WC{
	PS::S32 HydrDispatchKernel(const PS::S32, const PS::S32, const STD::EPI::Hydro**, const PS::S32*, const STD::EPJ::Hydro**, const PS::S32*);
	PS::S32 HydrRetrieveKernel(const PS::S32, const PS::S32, const PS::S32*, STD::RESULT::Hydro**);
};

