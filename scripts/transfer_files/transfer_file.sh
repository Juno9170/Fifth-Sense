read -p "Enter source path: " source
read -p "Enter destination path: " dest


tar --exclude='*.sh' -czf - "$source" | ssh root@qnxpi.local "tar -xzf - -C $dest"