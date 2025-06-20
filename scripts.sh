sudo apt update
sudo apt install -y build-essential python3-dev python3-pip libopenmpi-dev openmpi-bin python3-setuptools
sudo apt install -y mpich
pip3 install mpi4py numpy matplotlib

echo "MPICH, Python, and required Python libraries have been installed."

# mpirun --mca btl_tcp_if_include eno1 --hostfile hostfile -np 8 python3 ass2.py Isabel_1000x1000x200_float32.raw 2 2 2 0.5 opacity_tf.txt color_tf.txt