Checkout SimLOD branch IO: https://github.com/m-schuetz/SimLOD/tree/io
```
git checkout git@github.com:m-schuetz/SimLOD.git -b IO --track origin/IO
```

Create solution with cmake:
```
mkdir build
cd build
cmake ../
```

Open solution `SimLOD.sln`, edit `main_create_training.cpp`,
around line 40, change the input path like:
```
string pointcloud_dir = "E:\\datasets\\point_clouds\\CA13_las";
```

This will extract ground-truth heightmaps and textures from LAS point clouds.
