sed -i s/fillmein/Rapids/g settings.ini
nbdev_build_lib
sed -i s/Rapids/Pytorch-Classification/g settings.ini
nbdev_build_lib
sed -i s/Pytorch-Classification/Tabular/g settings.ini
nbdev_build_lib
sed -i s/Tabular/fillmein/g settings.ini
