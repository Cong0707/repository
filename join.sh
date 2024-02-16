sudo curl -s https://install.zerotier.com/ | sudo bash
sudo zerotier-cli join 6ab565387a0d0396
sudo curl -sfL https://rancher-mirror.rancher.cn/k3s/k3s-install.sh | INSTALL_K3S_MIRROR=cn K3S_URL=https://172.30.25.108:6443 K3S_TOKEN="K10641a92a44256ef9dd52cf711091c1ef9a0ac392c49107cfdcb1d541a4e2dd76f::server:315d5beb2391440a5acd97e2f78751f8" sh -
