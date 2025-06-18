@echo off
echo "Converting LAZ to LAS"
echo "Install LAStools https://rapidlasso.de/product-overview/"
echo "Change the path in this script to the path of the LAStools bin folder if it is different from C:\Program Files\LAStools\bin\las2las.exe"
set /p path="Enter the path of the folder containing LAZ files: "
cd %path%
"C:\Program Files\LAStools\bin\las2las.exe" -i *.laz -olas
echo "LAS files are written there"