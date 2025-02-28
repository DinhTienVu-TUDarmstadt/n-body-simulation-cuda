#include "naive_cuda_simulation.cuh"
#include "physics/gravitation.h"
#include "physics/mechanics.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "cuda_wrappers.cuh"
#include "constants.h"

void NaiveCudaSimulation::allocate_device_memory(Universe& universe, void** d_weights, void** d_forces, void** d_velocities, void** d_positions){
    cudaMalloc(d_weights, universe.num_bodies * sizeof(double));
    cudaMalloc(d_forces, universe.num_bodies * sizeof(double2));
    cudaMalloc(d_velocities, universe.num_bodies * sizeof(double2));
    cudaMalloc(d_positions, universe.num_bodies * sizeof(double2));
}

void NaiveCudaSimulation::free_device_memory(void** d_weights, void** d_forces, void** d_velocities, void** d_positions){
    cudaFree(*d_weights);
    cudaFree(*d_forces);
    cudaFree(*d_velocities);
    cudaFree(*d_positions);

    *d_weights = nullptr;
    *d_forces = nullptr;
    *d_velocities = nullptr;
    *d_positions = nullptr;
}

void NaiveCudaSimulation::copy_data_to_device(Universe& universe, void* d_weights, void* d_forces, void* d_velocities, void* d_positions){
    std::uint32_t num_bodies = universe.num_bodies;

    parprog_cudaMemcpy(d_weights, universe.weights.data(), num_bodies * sizeof(double), cudaMemcpyHostToDevice);

    std::vector<double2> forces(num_bodies);
    std::vector<double2> velocities(num_bodies);
    std::vector<double2> positions(num_bodies);

    for (std::uint32_t i = 0; i < num_bodies; i++) {
        forces[i] = { universe.forces[i][0], universe.forces[i][1] };
        velocities[i] = { universe.velocities[i][0], universe.velocities[i][1] };
        positions[i] = { universe.positions[i][0], universe.positions[i][1] };
    }

    parprog_cudaMemcpy(d_forces, forces.data(), num_bodies * sizeof(double2), cudaMemcpyHostToDevice);
    parprog_cudaMemcpy(d_velocities, velocities.data(), num_bodies * sizeof(double2), cudaMemcpyHostToDevice);
    parprog_cudaMemcpy(d_positions, positions.data(), num_bodies * sizeof(double2), cudaMemcpyHostToDevice);
}

void NaiveCudaSimulation::copy_data_from_device(Universe& universe, void* d_weights, void* d_forces, void* d_velocities, void* d_positions){
    std::size_t num_bodies = universe.num_bodies;

    parprog_cudaMemcpy(universe.weights.data(), d_weights, num_bodies * sizeof(double), cudaMemcpyDeviceToHost);

    std::vector<double2> forces(num_bodies);
    std::vector<double2> velocities(num_bodies);
    std::vector<double2> positions(num_bodies);

    parprog_cudaMemcpy(forces.data(), d_forces, num_bodies * sizeof(double2), cudaMemcpyDeviceToHost);
    parprog_cudaMemcpy(velocities.data(), d_velocities, num_bodies * sizeof(double2), cudaMemcpyDeviceToHost);
    parprog_cudaMemcpy(positions.data(), d_positions, num_bodies * sizeof(double2), cudaMemcpyDeviceToHost);

    for (std::size_t i = 0; i < num_bodies; i++) {
        universe.forces[i] = { forces[i].x, forces[i].y };
        universe.velocities[i] = { velocities[i].x, velocities[i].y };
        universe.positions[i] = { positions[i].x, positions[i].y };
    }
}
__device__ double2 operator-(const double2& a, const double2& b) {
    double2 result;
    result.x = a.x - b.x;
    result.y = a.y - b.y;
    return result;
}
__global__
void calculate_forces_kernel(std::uint32_t num_bodies, double2* d_positions, double* d_weights, double2* d_forces){
    std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < num_bodies) {
        double2 force = {0.0, 0.0};

        for (std::uint32_t j = 0; j < num_bodies; j++) {
            if (i != j) {
                double2 position_i = d_positions[i];
                double2 position_j = d_positions[j];

                double dx = position_j.x - position_i.x;
                double dy = position_j.y - position_i.y;
                double distance_squared = dx * dx + dy * dy;

                if (distance_squared > 0.0) {
                    double distance = sqrt(distance_squared);
                    double force_magnitude = (6.67430*1e-11 * d_weights[i] * d_weights[j]) / distance_squared;

                    // Tính toán hướng của lực tác dụng từ j lên i
                    double2 direction = {dx / distance, dy / distance};

                    // Cộng dồn lực vào tổng lực của cơ thể i
                    force.x += force_magnitude * direction.x;
                    force.y += force_magnitude * direction.y;
                }
            }
        }

        d_forces[i] = force;
    }
}

void NaiveCudaSimulation::calculate_forces(Universe& universe, void* d_positions, void* d_weights, void* d_forces){
    std::uint32_t num_bodies = universe.num_bodies;
    std::uint32_t block_size = 256;
    std::uint32_t grid_size = (num_bodies + block_size - 1) / block_size;

    calculate_forces_kernel<<<grid_size, block_size>>>(num_bodies, (double2*)d_positions, (double*)d_weights, (double2*)d_forces);

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
}

__global__
void calculate_velocities_kernel(std::uint32_t num_bodies, double2* d_forces, double* d_weights, double2* d_velocities){
    std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < num_bodies) {
        double2 force_i = d_forces[i];
        double mass_i = d_weights[i];

        double2 acceleration_i = {force_i.x / mass_i, force_i.y / mass_i};

        double2 velocity_i = d_velocities[i];

        double t = 2.628e+6;

        double2 new_velocity = {velocity_i.x + acceleration_i.x * t,
                                velocity_i.y + acceleration_i.y * t};

        d_velocities[i] = new_velocity;
    }
}

void NaiveCudaSimulation::calculate_velocities(Universe& universe, void* d_forces, void* d_weights, void* d_velocities){
    std::uint32_t num_bodies = universe.num_bodies;

    std::uint32_t block_size = 256;
    std::uint32_t grid_size = (num_bodies + block_size - 1) / block_size;

    calculate_velocities_kernel<<<grid_size, block_size>>>(num_bodies, (double2*)d_forces, (double*)d_weights, (double2*)d_velocities);

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
}

__global__
void calculate_positions_kernel(std::uint32_t num_bodies, double2* d_velocities, double2* d_positions){
    std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < num_bodies) {
        double2 position_i = d_positions[i];
        double2 velocity_i = d_velocities[i];

        double t = 2.628e+6;
        double2 displacement_i = {velocity_i.x * t, velocity_i.y * t};

        double2 new_position = {position_i.x + displacement_i.x, position_i.y + displacement_i.y};

        d_positions[i] = new_position;
    }
}

void NaiveCudaSimulation::calculate_positions(Universe& universe, void* d_velocities, void* d_positions){
    std::uint32_t num_bodies = universe.num_bodies;

    std::uint32_t block_size = 256;
    std::uint32_t grid_size = (num_bodies + block_size - 1) / block_size;

    calculate_positions_kernel<<<grid_size, block_size>>>(num_bodies, (double2*)d_velocities, (double2*)d_positions);

    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error in calculate_positions kernel: " << cudaGetErrorString(error) << std::endl;
    }
}

void NaiveCudaSimulation::simulate_epochs(Plotter& plotter, Universe& universe, std::uint32_t num_epochs, bool create_intermediate_plots, std::uint32_t plot_intermediate_epochs){

    void* d_weights = nullptr;
    void* d_forces = nullptr;
    void* d_velocities = nullptr;
    void* d_positions = nullptr;

    // Allocate memory on the GPU for the necessary variables
    cudaMalloc(&d_weights, universe.num_bodies * sizeof(std::uint32_t));  // Adjust size based on actual data
    cudaMalloc(&d_forces, universe.num_bodies * sizeof(float) * 3);        // Assuming 3D forces
    cudaMalloc(&d_velocities, universe.num_bodies * sizeof(float) * 3);    // Assuming 3D velocities
    cudaMalloc(&d_positions, universe.num_bodies * sizeof(float) * 3);     // Assuming 3D positions

    // Iterate over epochs
    for (std::uint32_t epoch = 0; epoch < num_epochs; ++epoch) {
        // Call simulate_epoch for each epoch
        simulate_epoch(plotter, universe, create_intermediate_plots, plot_intermediate_epochs, d_weights, d_forces, d_velocities, d_positions);

        // Update the universe state if necessary (this can be done after each epoch if needed)
    }

    // Free GPU memory after all epochs are done
    cudaFree(d_weights);
    cudaFree(d_forces);
    cudaFree(d_velocities);
    cudaFree(d_positions);
}

__global__
void get_pixels_kernel(std::uint32_t num_bodies, double2* d_positions, std::uint8_t* d_pixels, std::uint32_t plot_width, std::uint32_t plot_height, double plot_bounding_box_x_min, double plot_bounding_box_x_max, double plot_bounding_box_y_min, double plot_bounding_box_y_max){
    std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < num_bodies) {
        double2 position = d_positions[i];

        if (position.x >= plot_bounding_box_x_min && position.x <= plot_bounding_box_x_max &&
            position.y >= plot_bounding_box_y_min && position.y <= plot_bounding_box_y_max) {

            std::uint32_t pixel_x = (position.x - plot_bounding_box_x_min) * (plot_width - 1) / (plot_bounding_box_x_max - plot_bounding_box_x_min);
            std::uint32_t pixel_y = (position.y - plot_bounding_box_y_min) * (plot_height - 1) / (plot_bounding_box_y_max - plot_bounding_box_y_min);

            pixel_x = min(pixel_x, plot_width - 1);
            pixel_y = min(pixel_y, plot_height - 1);

            std::uint32_t pixel_index = pixel_y * plot_width + pixel_x;

            d_pixels[pixel_index] = 255;
            }
    }
}

std::vector<std::uint8_t> NaiveCudaSimulation::get_pixels(std::uint32_t plot_width, std::uint32_t plot_height, BoundingBox plot_bounding_box, void* d_positions, std::uint32_t num_bodies){
    std::vector<std::uint8_t> pixels;

    std::uint8_t* d_pixels = nullptr;
    cudaMalloc((void**)&d_pixels, plot_width * plot_height * sizeof(std::uint8_t));

    dim3 blockDim(256);
    dim3 gridDim((num_bodies + blockDim.x - 1) / blockDim.x);

    get_pixels_kernel<<<gridDim, blockDim>>>(num_bodies, (double2*) d_positions, d_pixels, plot_width, plot_height, plot_bounding_box.x_min, plot_bounding_box.x_max, plot_bounding_box.y_min, plot_bounding_box.y_max);
    cudaDeviceSynchronize();
    cudaMemcpy(pixels.data(), d_pixels, pixels.size() * sizeof(std::uint8_t), cudaMemcpyDeviceToHost);
    cudaFree(d_pixels);
    return pixels;
}

__global__
void compress_pixels_kernel(std::uint32_t num_raw_pixels, std::uint8_t* d_raw_pixels, std::uint8_t* d_compressed_pixels){
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_raw_pixels / 8)
    {
        std::uint8_t compressed_value = 0;

        for (std::uint32_t i = 0; i < 8; ++i)
        {
            std::uint32_t raw_pixel_idx = idx * 8 + i;

            std::uint8_t pixel_value = d_raw_pixels[raw_pixel_idx];

            if (pixel_value != 0)
            {
                compressed_value |= (1 << i);
            }
        }

        d_compressed_pixels[idx] = compressed_value;
    }
}

void NaiveCudaSimulation::compress_pixels(std::vector<std::uint8_t>& raw_pixels, std::vector<std::uint8_t>& compressed_pixels){
    if (compressed_pixels.size() < raw_pixels.size() / 8) {
        compressed_pixels.resize(raw_pixels.size() / 8);
    }

    std::uint8_t* d_raw_pixels;
    std::uint8_t* d_compressed_pixels;
    cudaMalloc(&d_raw_pixels, raw_pixels.size() * sizeof(std::uint8_t));
    cudaMalloc(&d_compressed_pixels, compressed_pixels.size() * sizeof(std::uint8_t));

    cudaMemcpy(d_raw_pixels, raw_pixels.data(), raw_pixels.size() * sizeof(std::uint8_t), cudaMemcpyHostToDevice);


    dim3 blockDim(256);
    dim3 gridDim((raw_pixels.size() / 8 + blockDim.x - 1) / blockDim.x);


    compress_pixels_kernel<<<gridDim, blockDim>>>(raw_pixels.size(), d_raw_pixels, d_compressed_pixels);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in compress_pixels_kernel: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    cudaMemcpy(compressed_pixels.data(), d_compressed_pixels, compressed_pixels.size() * sizeof(std::uint8_t), cudaMemcpyDeviceToHost);

    cudaFree(d_raw_pixels);
    cudaFree(d_compressed_pixels);
}

void NaiveCudaSimulation::simulate_epoch(Plotter& plotter, Universe& universe, bool create_intermediate_plots, std::uint32_t plot_intermediate_epochs, void* d_weights, void* d_forces, void* d_velocities, void* d_positions){
    calculate_forces(universe, d_positions, d_weights, d_forces);
    calculate_velocities(universe, d_forces, d_weights, d_velocities);
    calculate_positions(universe, d_velocities, d_positions);

    universe.current_simulation_epoch++;
    if(create_intermediate_plots){
        if(universe.current_simulation_epoch % plot_intermediate_epochs == 0){
            std::vector<std::uint8_t> pixels = get_pixels(plotter.get_plot_width(), plotter.get_plot_height(), plotter.get_plot_bounding_box(), d_positions, universe.num_bodies);
            plotter.add_active_pixels_to_image(pixels);

            // This is a dummy to use compression in plotting, although not beneficial performance-wise
            // ----
            // std::vector<std::uint8_t> compressed_pixels;
            // compressed_pixels.resize(pixels.size()/8);
            // compress_pixels(pixels, compressed_pixels);
            // plotter.add_compressed_pixels_to_image(compressed_pixels);
            // ----

            plotter.write_and_clear();
        }
    }
}

void NaiveCudaSimulation::calculate_forces_kernel_test_adapter(std::uint32_t grid_dim, std::uint32_t block_dim, std::uint32_t num_bodies, void* d_positions, void* d_weights, void* d_forces){
    // adapter function used by automatic tests. DO NOT MODIFY.
    dim3 blockDim(block_dim);
    dim3 gridDim(grid_dim);
    calculate_forces_kernel<<<gridDim, blockDim>>>(num_bodies, (double2*) d_positions, (double*) d_weights, (double2*) d_forces);
}

void NaiveCudaSimulation::calculate_velocities_kernel_test_adapter(std::uint32_t grid_dim, std::uint32_t block_dim, std::uint32_t num_bodies, void* d_forces, void* d_weights, void* d_velocities){
    // adapter function used by automatic tests. DO NOT MODIFY.
    dim3 blockDim(block_dim);
    dim3 gridDim(grid_dim);
    calculate_velocities_kernel<<<gridDim, blockDim>>>(num_bodies, (double2*) d_forces, (double*) d_weights, (double2*) d_velocities);
}

void NaiveCudaSimulation::calculate_positions_kernel_test_adapter(std::uint32_t grid_dim, std::uint32_t block_dim, std::uint32_t num_bodies, void* d_velocities, void* d_positions){
    // adapter function used by automatic tests. DO NOT MODIFY.
    dim3 blockDim(block_dim);
    dim3 gridDim(grid_dim);
    calculate_positions_kernel<<<gridDim, blockDim>>>(num_bodies, (double2*) d_velocities, (double2*) d_positions);
}
