/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

__device__ bool inverse3x3_dynamic(const float* A, float* Ainv, int N = 3) {
    // 增广矩阵，使用指针模拟动态行为
    float aug[3][6];
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            aug[i][j] = A[i * N + j];
        }
        for (int j = 0; j < N; ++j) {
            aug[i][j + N] = (i == j) ? 1.f : 0.f;
        }
    }

    // 高斯消元部分（动态 N 控制）
    for (int i = 0; i < N; ++i) {
        float pivot = aug[i][i];
        if (fabsf(pivot) < 1e-6f) return false;

        for (int j = 0; j < 2 * N; ++j) {
            aug[i][j] /= pivot;
        }

        for (int k = 0; k < N; ++k) {
            if (k == i) continue;
            float factor = aug[k][i];
            for (int j = 0; j < 2 * N; ++j) {
                aug[k][j] -= factor * aug[i][j];
            }
        }
    }

    // 拷贝逆矩阵部分
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            Ainv[i * N + j] = aug[i][j + N];
        }
    }

    return true;
}


__device__ void compute1DGS(
    const float3& ray_origin,
    const float3& ray_dir,
    const float3& mu,
    const float3& scale,
	float mod,
	const float4& rot,
    float& mu_t,
    float& sigma_t)
{
	float3 delta = make_float3(ray_origin.x - mu.x,
                                ray_origin.y - mu.y,
                                ray_origin.z - mu.z);
	// 1. 计算 Sigma
	float Sigma[9];
	computeSigma(Sigma, scale, mod, rot);  // 这是你的已有函数

	// 2. 使用通用矩阵求逆（慢）
	float invSigma[9];
	int dynamic_N = threadIdx.x % 5 + 1;  //
	dynamic_N = 3;  // 
	bool ok = inverse3x3_dynamic(Sigma, invSigma,dynamic_N);
	if (!ok) {
		// 处理不可逆情况
		mu_t = 0.0;
    	sigma_t = 0.0;
	}

	// 3. 用 invSigma 计算 A 和 B
	float3 Su = mat3x3TimesVec3(ray_dir, invSigma);
	float3 Sd = mat3x3TimesVec3(delta, invSigma);

	float A = float3dot(Su, ray_dir);
	float B = float3dot(Sd, ray_dir);

    mu_t = -B / A;
    sigma_t = 1.f / sqrtf(A);
}

__device__ float interval_probability_erf(float mean, float sigma, float a, float b) {
    // in case of 0
    if (sigma <= 1e-6f) return (a <= mean && mean <= b) ? 1.0f : 0.0f;

    float sqrt2 = 1.4142135623730951f;  // CUDART_SQRT_TWO
    float z1 = (a - mean) / (sigma * sqrt2);
    float z2 = (b - mean) / (sigma * sqrt2);

    float result = 0.5 * (erf(z2) - erf(z1));
    return result;
}

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

// Forward version of 2D covariance matrix computation
__device__ float3 computeCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	float3 t = transformPoint4x3(mean, viewmatrix);

	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	glm::mat3 T = W * J;

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Apply low-pass filter: every Gaussian should be at least
	// one pixel wide/high. Discard 3rd row and column.
	cov[0][0] += 0.3f;
	cov[1][1] += 0.3f;
	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
	// Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	// Normalize quaternion to get valid rotation
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 M = S * R;

	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;

	// Covariance is symmetric, only store upper right
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}

// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessCUDA(int P, int D, int M,
	const float* orig_points,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	int* radii,
	float2* points_xy_image,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered,
	bool rr_clipping,
	int n_clips ,
	float* clippers
)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	// Perform near culling, quit if outside.
	float3 p_view;
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;

	

	// Transform point by projecting
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };

	// perfrom the clippping here
	/**
	The clipping logic
	1. check how many clippers we have
	2. start one by one
	3. check wherther use rr_clipping
		1. if yes, use 3 time scales to judge whether render
		2. if false, use mean to judge whether render
	*/
	if (n_clips>0){
		for (int clps = 0; clps < n_clips; clps++){
			float3 p_normal = { clippers[4 * clps], clippers[4 * clps + 1], clippers[4 * clps + 2] };
			float p_dis= clippers[4* clps+3];
			if(rr_clipping){

				bool vis_flag=is_visible(p_orig, p_normal, p_dis);
				if (!vis_flag){
					// if we use rr_clipping use 3*scales to judge the distance, for those in invisible point but cutoff
					float point2plane_dis=point_to_plane_distance(p_orig, p_normal, p_dis);
					float sacle_x=scales[idx].x*scale_modifier;
					float sacle_y=scales[idx].y*scale_modifier;
					float sacle_z=scales[idx].z*scale_modifier;
					float scale_max= 3.0f*fmaxf(fmaxf(sacle_x, sacle_y), sacle_z);
					if(point2plane_dis>scale_max) return; // only ignore when the point in invisible area and too far from the clipping plane
				}
			}else{
				// use the mean to judge the visible
				bool vis_flag=is_visible(p_orig, p_normal, p_dis);
				if (!vis_flag){
					return;// directly return, no render later
				}
			}
		}
	}




	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

	// If 3D covariance matrix is precomputed, use it, otherwise compute
	// from scaling and rotation parameters. 
	const float* cov3D;
	if (cov3D_precomp != nullptr)
	{
		cov3D = cov3D_precomp + idx * 6;
	}
	else
	{
		computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
		cov3D = cov3Ds + idx * 6;
	}

	// Compute 2D screen-space covariance matrix
	float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);

	// Invert covariance (EWA algorithm)
	float det = (cov.x * cov.z - cov.y * cov.y);
	if (det == 0.0f)
		return;
	float det_inv = 1.f / det;
	float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

	// Compute extent in screen space (by finding eigenvalues of
	// 2D covariance matrix). Use extent to compute a bounding rectangle
	// of screen-space tiles that this Gaussian overlaps with. Quit if
	// rectangle covers 0 tiles. 
	float mid = 0.5f * (cov.x + cov.z);
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
	float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
	float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
	uint2 rect_min, rect_max;
	getRect(point_image, my_radius, rect_min, rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	// If colors have been precomputed, use them, otherwise convert
	// spherical harmonics coefficients to RGB color.
	if (colors_precomp == nullptr)
	{
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}

	// Store some useful helper data for the next steps.
	depths[idx] = p_view.z;
	radii[idx] = my_radius;
	points_xy_image[idx] = point_image;
	// Inverse 2D covariance and opacity neatly pack into one float4
	conic_opacity[idx] = { conic.x, conic.y, conic.z, opacities[idx] };
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float* __restrict__ depths,
	const float4* __restrict__ conic_opacity,
	float* __restrict__ out_alpha,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color,
	float* __restrict__ out_depth,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	const glm::vec3* __restrict__ cam_pos,
	const float* __restrict__ orig_points,
	const glm::vec3* __restrict__ scales,
	const float scale_modifier,
	const glm::vec4* __restrict__ rotations,
	const float* __restrict__ viewmatrix,
	const float* __restrict__ projmatrix,
	float* __restrict__ debug_values,
	bool rr_clipping,
	bool rr_strategy,
	bool oenD_gs_strategy,
	int n_clips ,
	const float* clippers,
	bool vizPlane,
	int n_inters ,
	const float* intersections_tensor
)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;


	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];

	__shared__ float3 collected_means[BLOCK_SIZE];
	__shared__ float3 collected_scales[BLOCK_SIZE];
	__shared__ float4 collected_rots[BLOCK_SIZE];


	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0 };
	float D = 0;

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];


			collected_means[block.thread_rank()]={orig_points[3 * coll_id], orig_points[3 * coll_id + 1], orig_points[3 * coll_id + 2]};
			collected_scales[block.thread_rank()]={scales[coll_id].x,scales[coll_id].y,scales[coll_id].z};
			collected_rots[block.thread_rank()]={rotations[coll_id].x,rotations[coll_id].y,rotations[coll_id].z,rotations[coll_id].w};
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;

			// Resample using conic matrix (cf. "Surface 
			// Splatting" by Zwicker et al., 2001)
			float2 xy = collected_xy[j];
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			float4 con_o = collected_conic_opacity[j];
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			float decay_weight=0.0f;// decay the power, I need  to set it as 0 for initilization, otherwise have some werid case for the intersection! need to fix
			if (power > 0.0f) continue;


			// DL: Here I need to implement the decaying weights logic, then we can optimize it
			/*
			Here we say, we have the plane in the world with the normal pointing to x+, d=0

			# First, we use ray tracing to calculate the power instead of the distribution

			*/
			if (n_clips>0){
				for (int clps = 0; clps < n_clips; clps++){
					float3 p_normal = { clippers[4 * clps], clippers[4 * clps + 1], clippers[4 * clps + 2] };
					float p_dis= clippers[4* clps+3];
					if(rr_clipping){
						// now all the points are kind of visible, only for those close to the plane as 3*scales,start ray tracing
						float3 p_orig=collected_means[j];
						//only perform ray tracing for the cutoff gaussian
						// if we use rr_clipping use 3*scales to judge the distance, for those in invisible point but cutoff
						float point2plane_dis=point_to_plane_distance(p_orig, p_normal, p_dis);
						float sacle_x=collected_scales[j].x*scale_modifier;
						float sacle_y=collected_scales[j].y*scale_modifier;
						float sacle_z=collected_scales[j].z*scale_modifier;
						float scale_max= 3.0f*fmaxf(fmaxf(sacle_x, sacle_y), sacle_z);
						if(rr_strategy&&(point2plane_dis<scale_max)){// remove this if we want to compare the RaRa strategy if(point2plane_dis<scale_max){// remove this if we want to compare the RaRa strategy
							// for those close to the plane as 3 times scale, we nned the ray tracing to compute decaying weights
							// 1. calculate the ray and calculate the ray intersections compute the length as the weight
							// generate ray from pixel
							float x_ndc_back = (pixf.x) / float(W) * 2 - 1; // be careful about the 0.5 shift
							float y_ndc_back = (pixf.y) / float(H) * 2 - 1;

							float x_dir = x_ndc_back * tan_fovx;
							float y_dir = y_ndc_back * tan_fovy;
							float3 ray_dir_cam=normalize_float3({ x_dir, y_dir, float(1.0)});
							float3 ray_dir_world;
							float3 ray_origin=make_float3(cam_pos->x, cam_pos->y, cam_pos->z);// cam_pos is a pointer here
							float3 e_hit_min={-10.0f,-10.0f,-10.0f};
							float3 e_hit_max={-10.0f,-10.0f,-10.0f};
							//float3 ellipsoid_mean_w=collected_means[j];
							// we need to make sure all computation is on world space
							// trasform ray_dir from camera to world space
							float Cam_R_T[9];// store the camera to world transform,store as row-major
							float T_mat[9]; // store the matrix converting ellipsord to sphere

							/**
							cam_R_T is the transpose of world to camera
							*/
							Cam_R_T[0]=viewmatrix[0]; Cam_R_T[1]=viewmatrix[1]; Cam_R_T[2]=viewmatrix[2];
							Cam_R_T[3]=viewmatrix[4]; Cam_R_T[4]=viewmatrix[5]; Cam_R_T[5]=viewmatrix[6];
							Cam_R_T[6]=viewmatrix[8]; Cam_R_T[7]=viewmatrix[9]; Cam_R_T[8]=viewmatrix[10];

							ray_dir_world=mat3x3TimesVec3(ray_dir_cam,Cam_R_T);
							computeT_mat(T_mat,collected_scales[j], scale_modifier, collected_rots[j]);
							float3 ray_translated={ray_origin.x-collected_means[j].x,ray_origin.y-collected_means[j].y,ray_origin.z-collected_means[j].z};
							float3 e_rel=mat3x3TimesVec3(ray_translated,T_mat);
							float3 d_rel =mat3x3TimesVec3(ray_dir_world,T_mat);

							float a=float3dot(d_rel,d_rel);
							float b=2*float3dot(e_rel,d_rel);
							float c=float3dot(e_rel, e_rel) - 1;
							float disc = b*b - 4*a*c;
							if (disc > 0.0f) { 
								// only when we have intersection we will continue
								float sqrt_disc = sqrt(disc);
								float t0 = (-b - sqrt_disc) / (2 * a);
								float t1 = (-b + sqrt_disc) / (2 * a);
								float e_t_min=-10.0f;
								float e_t_max=-10.0f;
								e_t_min=min(t0,t1);
								e_t_max=max(t0,t1);
								float3 plane_normal=p_normal;
								float plane_d=p_dis;
								float t_p=intersect_ray_plane(ray_origin,ray_dir_world,plane_normal,plane_d); 
								
								if (e_t_min > 0.0f && e_t_max>0.0f){// two intersection, we keep the weights
									e_hit_min=ray_origin+ray_dir_world*e_t_min;
									e_hit_max=ray_origin+ray_dir_world*e_t_max;
									bool e_hit_min_vis=is_visible(e_hit_min, plane_normal, plane_d);
									bool e_hit_max_vis=is_visible(e_hit_max, plane_normal, plane_d);
									float3 diff_vec=e_hit_max-e_hit_min;
									float hit_norm=length(diff_vec);
									if (e_hit_min_vis&&e_hit_max_vis){
										decay_weight=1.0f;
									}else if(t_p>0.0f){
										float3 p_hit=ray_origin+ray_dir_world*t_p;
										if (e_hit_min_vis){
											if (oenD_gs_strategy){
												float mu=0.0;
												float sigma=0.0;
												// compute1DGS(
												// 			ray_origin,
												// 			ray_dir_world,
												// 			p_orig,
												// 			T_mat,
												// 			mu,
												// 			sigma);
												compute1DGS(
															ray_origin,
															ray_dir_world,
															p_orig,
															collected_scales[j], scale_modifier, collected_rots[j],
															mu,
															sigma);
												
												decay_weight=min(0.99f, interval_probability_erf(mu,sigma, e_t_min, t_p));

											}else{
												float vis_norm=length(e_hit_min - p_hit);
												decay_weight=vis_norm/hit_norm;
											}

										}else if (e_hit_max_vis){
											if(oenD_gs_strategy){
												float mu=0.0;
												float sigma=0.0;
												// compute1DGS(
												// 			ray_origin,
												// 			ray_dir_world,
												// 			p_orig,
												// 			T_mat,
												// 			mu,
												// 			sigma);
												compute1DGS(
															ray_origin,
															ray_dir_world,
															p_orig,
															collected_scales[j], scale_modifier, collected_rots[j],
															mu,
															sigma);
												decay_weight=min(0.99f, interval_probability_erf(mu,sigma,t_p, e_t_max));
											}else{
												float vis_norm=length(e_hit_max - p_hit);
												decay_weight=vis_norm/hit_norm;
											}
										}
									}else{
										decay_weight=0.0f;
									}
								} else{
									// to do : fix the situation when we have the camera inside one gaussian
								}
								
								//take some values out for debugging
								/** 
								if (((pixf.x-float(W/2)-1)<0.5)&&((pixf.x-float(W/2))>=0.0)&&((pixf.y-float(H/2))<0.5)&&((pixf.y-float(H/2))>=0.0)){
									// we only save the center data
									debug_values[0]=pixf.x;
									debug_values[1]=pixf.y;
								}
									*/
							}// no intersection keep rasterization
						}else if(!rr_strategy){ // ray tracing all ellipsoid
							// for those close to the plane as 3 times scale, we nned the ray tracing to compute decaying weights
							// 1. calculate the ray and calculate the ray intersections compute the length as the weight
							// generate ray from pixel
							float x_ndc_back = (pixf.x) / float(W) * 2 - 1; // be careful about the 0.5 shift
							float y_ndc_back = (pixf.y) / float(H) * 2 - 1;

							float x_dir = x_ndc_back * tan_fovx;
							float y_dir = y_ndc_back * tan_fovy;
							float3 ray_dir_cam=normalize_float3({ x_dir, y_dir, float(1.0)});
							float3 ray_dir_world;
							float3 ray_origin=make_float3(cam_pos->x, cam_pos->y, cam_pos->z);// cam_pos is a pointer here
							float3 e_hit_min={-10.0f,-10.0f,-10.0f};
							float3 e_hit_max={-10.0f,-10.0f,-10.0f};
							//float3 ellipsoid_mean_w=collected_means[j];
							// we need to make sure all computation is on world space
							// trasform ray_dir from camera to world space
							float Cam_R_T[9];// store the camera to world transform,store as row-major
							float T_mat[9]; // store the matrix converting ellipsord to sphere

							/**
							cam_R_T is the transpose of world to camera
							*/
							Cam_R_T[0]=viewmatrix[0]; Cam_R_T[1]=viewmatrix[1]; Cam_R_T[2]=viewmatrix[2];
							Cam_R_T[3]=viewmatrix[4]; Cam_R_T[4]=viewmatrix[5]; Cam_R_T[5]=viewmatrix[6];
							Cam_R_T[6]=viewmatrix[8]; Cam_R_T[7]=viewmatrix[9]; Cam_R_T[8]=viewmatrix[10];

							ray_dir_world=mat3x3TimesVec3(ray_dir_cam,Cam_R_T);
							computeT_mat(T_mat,collected_scales[j], scale_modifier, collected_rots[j]);
							float3 ray_translated={ray_origin.x-collected_means[j].x,ray_origin.y-collected_means[j].y,ray_origin.z-collected_means[j].z};
							float3 e_rel=mat3x3TimesVec3(ray_translated,T_mat);
							float3 d_rel =mat3x3TimesVec3(ray_dir_world,T_mat);

							float a=float3dot(d_rel,d_rel);
							float b=2*float3dot(e_rel,d_rel);
							float c=float3dot(e_rel, e_rel) - 1;
							float disc = b*b - 4*a*c;
							if (disc > 0.0f) { 
								// only when we have intersection we will continue
								float sqrt_disc = sqrt(disc);
								float t0 = (-b - sqrt_disc) / (2 * a);
								float t1 = (-b + sqrt_disc) / (2 * a);
								float e_t_min=-10.0f;
								float e_t_max=-10.0f;
								e_t_min=min(t0,t1);
								e_t_max=max(t0,t1);
								float3 plane_normal=p_normal;
								float plane_d=p_dis;
								float t_p=intersect_ray_plane(ray_origin,ray_dir_world,plane_normal,plane_d); 
								
								if (e_t_min > 0.0f && e_t_max>0.0f){// two intersection, we keep the weights
									e_hit_min=ray_origin+ray_dir_world*e_t_min;
									e_hit_max=ray_origin+ray_dir_world*e_t_max;
									bool e_hit_min_vis=is_visible(e_hit_min, plane_normal, plane_d);
									bool e_hit_max_vis=is_visible(e_hit_max, plane_normal, plane_d);
									float3 diff_vec=e_hit_max-e_hit_min;
									float hit_norm=length(diff_vec);
									if (e_hit_min_vis&&e_hit_max_vis){
										decay_weight=1.0f;
									}else if(t_p>0.0f){
										float3 p_hit=ray_origin+ray_dir_world*t_p;
										if (e_hit_min_vis){
											float vis_norm=length(e_hit_min - p_hit);
											decay_weight=vis_norm/hit_norm;
										}else if (e_hit_max_vis){
											float vis_norm=length(e_hit_max - p_hit);
											decay_weight=vis_norm/hit_norm;
										}
									}else{
										decay_weight=0.0f;
									}
								} else{
									// to do : fix the situation when we have the camera inside one gaussian
								}
								
								//take some values out for debugging
								/** 
								if (((pixf.x-float(W/2)-1)<0.5)&&((pixf.x-float(W/2))>=0.0)&&((pixf.y-float(H/2))<0.5)&&((pixf.y-float(H/2))>=0.0)){
									// we only save the center data
									debug_values[0]=pixf.x;
									debug_values[1]=pixf.y;
								}
									*/
							}// no intersection keep rasterization
						}else{
							decay_weight=1.0f;// keep original rasterization
						}
					}else{
						decay_weight=1.0f;// keep original rasterization
					}
				}
			}else{
				decay_weight=1.0f;// keep original rasterization
			}

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			// float alpha = min(0.99f, con_o.w *decay_weight);
			float alpha = min(0.99f, con_o.w * exp(power)*decay_weight);
			if (alpha < 1.0f / 255.0f)
				continue;
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}

			// Eq. (3) from 3D Gaussian splatting paper.
			float random=(float(collected_id[j])+100.0)/(float(collected_id[j])*2.0);
			// float3 fakeRGB={}

			for (int ch = 0; ch < CHANNELS; ch++)
					C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;
				
			D += depths[collected_id[j]] * alpha * T;
			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		out_alpha[pix_id] = 1 - T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
		out_depth[pix_id] = D;



		// render the line in the end, in case of the overwrite
		if (vizPlane){
			for(int i=0;i<n_inters-1;i++){
				float3 p_orig = { intersections_tensor[3 * i], intersections_tensor[3 * i + 1], intersections_tensor[3 * i + 2] };
				float4 p_hom = transformPoint4x4(p_orig, projmatrix);
				float p_w = 1.0f / (p_hom.w + 0.0000001f);
				float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };
				float2 point_image_start = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };

				int end=i+1;
				p_orig = { intersections_tensor[3 * end], intersections_tensor[3 * end + 1], intersections_tensor[3 * end + 2] };
				p_hom = transformPoint4x4(p_orig, projmatrix);
				p_w = 1.0f / (p_hom.w + 0.0000001f);
				p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };
				float2 point_image_end = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };

				if (pixel_on_line(pixf, point_image_start, point_image_end,0.5f)) {
					// 写入像素值，比如 framebuffer[pix_id] = 1;
					out_color[0 * H * W + pix_id]=1.0f;
					out_color[1 * H * W + pix_id]=0.0f;
					out_color[2 * H * W + pix_id]=0.0f;// fill red 
				}
			}
				// for the last segment to have the close rectangle
				int i=n_inters-1;
				float3 p_orig = { intersections_tensor[3 * i], intersections_tensor[3 * i + 1], intersections_tensor[3 * i + 2] };
				float4 p_hom = transformPoint4x4(p_orig, projmatrix);
				float p_w = 1.0f / (p_hom.w + 0.0000001f);
				float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };
				float2 point_image_start = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };

				int end=0;
				p_orig = { intersections_tensor[3 * end], intersections_tensor[3 * end + 1], intersections_tensor[3 * end + 2] };
				p_hom = transformPoint4x4(p_orig, projmatrix);
				p_w = 1.0f / (p_hom.w + 0.0000001f);
				p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };
				float2 point_image_end = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };

				if (pixel_on_line(pixf, point_image_start, point_image_end,0.5f)) {
					// 写入像素值，比如 framebuffer[pix_id] = 1;
					out_color[0 * H * W + pix_id]=1.0f;
					out_color[1 * H * W + pix_id]=0.0f;
					out_color[2 * H * W + pix_id]=0.0f;// fill red 
				}
		}

	}


}

void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float2* means2D,
	const float* colors,
	const float* depths,
	const float4* conic_opacity,
	float* out_alpha,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_color,
	float* out_depth,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	const glm::vec3* cam_pos,
	const float* orig_points,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* viewmatrix,
	const float* projmatrix,
	float* debug_values,
	bool rr_clipping,
	bool rr_strategy,
	bool oenD_gs_strategy,
	int n_clips ,
	float* clippers,
	bool vizPlane,
	int n_inters ,
	float* intersections_tensor
)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		means2D,
		colors,
		depths,
		conic_opacity,
		out_alpha,
		n_contrib,
		bg_color,
		out_color,
		out_depth,
		focal_x,focal_y,
		tan_fovx, tan_fovy,
		cam_pos,
		orig_points,
		scales,
		scale_modifier,
		rotations,
		viewmatrix,
		projmatrix,
		debug_values,
		rr_clipping,
		rr_strategy,
		oenD_gs_strategy,
		n_clips,
		clippers,
		vizPlane,
		n_inters ,
		intersections_tensor
	);
}

void FORWARD::preprocess(int P, int D, int M,
	const float* means3D,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	int* radii,
	float2* means2D,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered,
	bool rr_clipping,
	int n_clips ,
	float* clippers
)
{
	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P, D, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		shs,
		clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, 
		projmatrix,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		means2D,
		depths,
		cov3Ds,
		rgb,
		conic_opacity,
		grid,
		tiles_touched,
		prefiltered,
		rr_clipping,
		n_clips ,
		clippers
		);
}