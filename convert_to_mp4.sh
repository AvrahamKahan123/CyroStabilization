# google slides didn't like the AVI's codec, so converting
ffmpeg -i differences.avi -c:v libx264 -pix_fmt yuv420p differences.mp4