# leslie_mapping
Code for creating a 3D LIDAR map of Leslie and Highway 7.

Dependencies:

```
libnabo
libpointmatcher
Eigen 3.3
OpenCV 3.3
steam
```

Installing libnabo:

```
git clone https://github.com/ethz-asl/libnabo.git
cd libnabo && mkdir build && cd build && cmake .. && make && make install
```

Installing libpointmatcher:

```
git clone https://github.com/ethz-asl/libpointmatcher.git
cd libpointmatcher && mkdir build && cd build && cmake .. && make && make install
```

Installing steam:

```
mkdir \~/steam_ws && cd \~steam_ws
git clone https://github.com/utiasASRL/steam.git
cd steam && git submodule update --init --remote
cd deps/catkin && catkin build && cd ../.. && catkin build
```

Building this repo:

```
mkdir -p \~/map_ws/src/ && cd \~/map_ws/src
git clone https://github.com/keenan-burnett/leslie_mapping.git
cd ../.. && catkin init && catkin config --extend \~/steam_ws/devel/repo && catkin build
```
