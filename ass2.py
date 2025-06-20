from mpi4py import MPI
import numpy as np
import time
import sys
import matplotlib.pyplot as plt


def load_transfer_function(file_path, color):
    with open(file_path, 'r') as file:
        data = [float(val) for line in file for val in line.strip().replace(',', '').split()]

    if color:
        return [(data[i], tuple(data[i + 1:i + 4])) for i in range(0, len(data), 4)]
    else:
        return [(data[i], data[i + 1]) for i in range(0, len(data), 2)]

def interpolate_color(query_val, color_tf):
   
    for i in range(len(color_tf) - 1):
        x_start, color_start = color_tf[i]
        x_end, color_end = color_tf[i + 1]
        
        # Check if query_val is within the current range [x_start, x_end]
        if x_start <= query_val <= x_end:
            # Calculate interpolation factor (t) based on the relative position of query_val
            t = (query_val - x_start) / (x_end - x_start)
            
            # Interpolate the RGB values for each color channel (R, G, B)
            interpolated_color = [
                color_start[channel] + t * (color_end[channel] - color_start[channel])
                for channel in range(3)
            ]
            return interpolated_color

    # If query_val is out of the specified ranges, return black (default)
    return [0, 0, 0]


def interpolate_opacity(query_val, opacity_tf):
    # Loop through the opacity transformation points, excluding the last element
    for i in range(len(opacity_tf) - 1):
        x_start, y_start = opacity_tf[i]
        x_end, y_end = opacity_tf[i + 1]
        
        # Check if query_val is within the current range [x_start, x_end]
        if x_start <= query_val <= x_end:
            # Perform linear interpolation between the y-values based on the position of query_val
            t = (query_val - x_start) / (x_end - x_start)
            return y_start + t * (y_end - y_start)

    # If query_val is out of range, return 0 (default opacity)
    return 0

def interpolate_value(data, x, y, z):
    z_lower = int(z)
    z_upper = min(z_lower + 1, data.shape[2] - 1)
    
    # Calculate the interpolation factor based on z
    interpolation_factor = z - z_lower
    
    # Perform linear interpolation between the two z values
    value_at_z_lower = data[x, y, z_lower]
    value_at_z_upper = data[x, y, z_upper]
    
    interpolated_value = (1 - interpolation_factor) * value_at_z_lower + interpolation_factor * value_at_z_upper
    return interpolated_value

def ray_casting(subdomain, opacity_points, color_points, step_size):
    height, width, depth = subdomain.shape
    image = np.zeros((height, width, 3))
    for y in range(width):
        for x in range(height):
            accumulated_color = np.zeros(3)
            accumulated_opacity = 0
            z = 0.0
            while z < depth:
                data_val = interpolate_value(subdomain, x, y, z)
                color = np.array(interpolate_color(data_val, color_points))
                opacity = interpolate_opacity(data_val, opacity_points)

                accumulated_color += (1 - accumulated_opacity) * color * opacity
                accumulated_opacity += (1 - accumulated_opacity) * opacity

                if accumulated_opacity >= 0.98:
                    break

                z += step_size

            image[x, y, :] = accumulated_color
    return image



def merge_images(images, PX, PY, PZ):
    # Extract dimensions of individual images (assuming all images are the same size)
    height, width, _ = images[0].shape

    # Create an empty canvas for the final image with the desired total size
    merged_image = np.zeros((height * PX, width * PY, 3))

    # Iterate over the grid of subdomains (PX, PY)
    for x in range(PX):
        for y in range(PY):
            # Initialize empty arrays for accumulating color and opacity
            accumulated_color = np.zeros((height, width, 3))
            accumulated_opacity = np.zeros((height, width))

            # Process each depth slice (Z)
            for z in range(PZ):
                # Calculate the index for the current image in the list
                img_index = x * PY * PZ + y * PZ + z
                sub_image = images[img_index]  # Get the image at the current index

                # Calculate the remaining opacity and update color and opacity accumulations
                remaining_opacity = 1 - accumulated_opacity
                accumulated_color += sub_image * remaining_opacity[:, :, None]
                accumulated_opacity += remaining_opacity

            # Place the composited result into the corresponding location on the final image canvas
            merged_image[x * height:(x + 1) * height, 
                         y * width:(y + 1) * width, :] = accumulated_color

    return merged_image



def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    start_time_exec = MPI.Wtime()  # Start timing the total execution    

    if len(sys.argv) < 7:
        if rank == 0:
            print("Usage: mpirun -np <num_procs> python file_name.py <dataset_name> <PX> <PY> <PZ> <step_size> <opacity_tf> <color_tf>")
        sys.exit()

    dataset_name = sys.argv[1]
    PX = int(sys.argv[2])
    PY = int(sys.argv[3])
    PZ = int(sys.argv[4])
    step_size = float(sys.argv[5])
    opacity_tf_filename=sys.argv[6]

    color_tf_filename = sys.argv[7]
  
    if "1000x1000x200" not in dataset_name:
        raise ValueError("Dataset name doesn't match known dimensions.")
    shape=(1000,1000,200)
        
    if rank == 0:
        # Step 1: Load and distribute data
        start_time = MPI.Wtime()
        data = np.fromfile(dataset_name, dtype=np.float32).reshape(shape, order='F')

        # Step 2: Split the data along the first axis (x-axis) into `PX` subarrays
        x_slices = np.array_split(data, PX, axis=0)

         # Step 2: For each slice along the x-axis, split it along the second axis (y-axis) into `PY` subarrays
        subdomains = []
        for slice in x_slices:
           y_slices = np.array_split(slice, PY, axis=1)  # Split along y-axis
           subdomains.append(y_slices)
           
        # Send subdomains to all other ranks
        start_send_time = MPI.Wtime()
        for i in range(PX):
            for j in range(PY):
                for k in range(PZ):
                    target_rank = i * PY * PZ + j * PZ + k
                    subdomain_data = subdomains[i][j][:, :, k::PZ]

                    if target_rank == 0:
                        subdomain = subdomain_data  # Keep the subdomain for rank 0
                    else:
                        print(f"Sending subdomain to rank {target_rank} with shape {subdomain_data.shape}")
                        comm.send(subdomain_data, dest=target_rank, tag=target_rank)

        end_send_time = MPI.Wtime()
        data_distribution_time = end_send_time - start_send_time
        total_loading_time = MPI.Wtime() - start_time
        print(f"Data loading and distribution time: {total_loading_time:.4f} seconds")

    else:
        # Step 2: Receive subdomains on other ranks
        start_recv_time = MPI.Wtime()
        subdomain = comm.recv(source=0, tag=rank)
        end_recv_time = MPI.Wtime()
        recv_time = end_recv_time - start_recv_time
        print(f"Rank {rank} received subdomain with shape: {subdomain.shape}")
        
    max_send_time = comm.reduce(data_distribution_time, op=MPI.MAX, root=0)
    max_recv_time = comm.reduce(recv_time, op=MPI.MAX, root=0)
    
    # Step 3: Load transfer functions
    opacity_points = load_transfer_function(opacity_tf_filename, color=False)
    color_points = load_transfer_function(color_tf_filename, color=True)

    # Step 4: Perform ray casting (image generation)
    start_ray_casting_time = MPI.Wtime()
    img = ray_casting(subdomain, opacity_points, color_points, step_size)
    end_ray_casting_time = MPI.Wtime()
    ray_casting_time = end_ray_casting_time - start_ray_casting_time
    
    max_ray_casting_time = comm.reduce(ray_casting_time, op=MPI.MAX, root=0)
    # Step 5: Gather images from all ranks
    start_gather_time = MPI.Wtime()
    gathered_images = comm.gather(img, root=0)
    end_gather_time = MPI.Wtime()
    gather_time = end_gather_time - start_gather_time

    max_gather_time = comm.reduce(gather_time, op=MPI.MAX, root=0)

    if rank == 0:
        # Step 6: Merge gathered images and save final image
        merged_image = merge_images(gathered_images, PX, PY, PZ)
        output_filename = f"{PX}_{PY}_{PZ}_{step_size}.png"
        plt.imsave(output_filename, merged_image)

        print(f"Final image saved as: {output_filename}")
        print(f"Maximum ray casting time: {max_ray_casting_time:.4f} seconds")
        
        # Total communication time
        total_comm_time = max_send_time + max_recv_time + max_gather_time
        print(f"Total communication time: {total_comm_time:.6f} seconds")

    # Step 7: Calculate total execution time across all ranks
    total_execution_time = MPI.Wtime() - start_time_exec
    max_execution_time = comm.reduce(total_execution_time, op=MPI.MAX, root=0)

    if rank == 0:
        print(f"Total execution time: {max_execution_time:.4f} seconds")

if __name__ == "__main__":
    main()





# mpirun --mca btl_tcp_if_include eno1 --hostfile hostfile -np 8 python3 codee.py Isabel_1000x1000x200_float32.raw 2 2 2 0.5 opacity_TF.txt color_TF.txt